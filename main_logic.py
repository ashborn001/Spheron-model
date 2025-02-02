import config  # This will load and set environment variables

import os
from flask import jsonify
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langgraph.graph import START, StateGraph, END
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

import yaml
from jsonschema import validate, ValidationError

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)
absl.logging.set_stderrthreshold('info')




ICL_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "services": {"type": "object"},
        "global": {"type": "boolean"},
        "env": {"type": "object"},
        "profiles": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "mode": {"type": "string"},
                "duration": {"type": "string"},
                "tier": {"type": "string"},
            },
        },
        "compute": {"type": "object"},
        "placement": {"type": "object"},
        "pricing": {"type": "object"},
        "deployment": {"type": "object"},
    },
    "required": ["version", "services", "deployment"],
}


# Spheron YAML docs loader
url = "https://docs.spheron.network/user-guide/icl"
loader = RecursiveUrlLoader(url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text)
docs = loader.load()
concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in sorted(docs, key=lambda x: x.metadata["source"], reverse=True)])

# Define the output schema
class CodeOutput(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block excluding imports")

# Setup Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE},
)

structured_llm_gemini = llm.with_structured_output(CodeOutput, include_raw=True)

# Prompt to enforce tool use
code_gen_prompt_gemini = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """<instructions>  
You are a coding assistant with expertise in generating *Spheron deployment YAML configurations*.  
Your task is to generate a well-structured *YAML configuration* based on the given *Integration Composition Language (ICL) documentation*.

📌 *Rules to Follow*:
- *DO NOT change variable names* in the schema.
- *Modify only variable values* according to the given documentation.
- Ensure *proper YAML formatting* with correct indentation.
- The output should follow the *Schema Template* exactly for *single-service deployment*.

📖 *ICL Documentation Reference:*
-------
{context}
-------

### Schema Template (Single-Service Deployment)*
---
version: "<string>"
services:
  <service-name>:
    image: <string>
    expose:
      - port: <integer>
        as: <integer>
        to:
          - global: <boolean>
    env:
      - <key>=<value>
profiles:
  name: <string>
  mode: <string>
  duration: <string>
  tier: 
    - <string>
  compute:
    <service-name>:
      resources:
        cpu:
          units: <integer>
        memory:
          size: <string>
        storage:
          - size: <string>
        gpu:
          units: <integer>
          attributes:
            vendor: 
              <vendor-name>:
                - model: <string>
  placement:
    <location-name>:
      attributes:
        region: <string>
      pricing:
        <service-name>:
          token: <string>
          amount: <integer>
deployment:
  <service-name>:
    <location-name>:
      profile: <string>
      count: <integer>

---

🔍 *Task Instructions*:
1*Modify only the values* in the YAML using the provided documentation.  
2*Ensure all required fields are present* in the output.  
3*Strictly maintain variable names as they are in the schema*.  
4*Validate the YAML to match ICL specifications*.  
5*Return the generated YAML with proper structure*.

Respond only with the *final YAML configuration*.
    
    
    """

    ,
        ),
        ("user", "{messages}"),
    ]
)


# Optional: Check for errors in case tool use is flaky
def check_gemini_output(tool_output):
    """Check for parse error or failure to call the tool"""

    # Error with parsing
    if tool_output["parsing_error"]:
        # Report back output and parsing errors
        print("Parsing error!")
        raw_output = str(tool_output["raw"].content)
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )

    # Tool was not invoked
    elif not tool_output["parsed"]:
        print("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output


# Chain with output check
code_chain_gemini_raw = (
    code_gen_prompt_gemini | structured_llm_gemini | check_gemini_output
)


def insert_errors(inputs):
    """Insert errors for tool parsing in the messages"""

    # Get errors
    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "assistant",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "messages": messages,
        "context": inputs["context"],
    }


# This will be run as a fallback chain
fallback_chain = insert_errors | code_chain_gemini_raw
N = 4  # Max re-tries
code_gen_chain_re_try = code_chain_gemini_raw.with_fallbacks(
    fallbacks=[fallback_chain] * N, exception_key="error"
)


def parse_output(solution):
    """When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'."""

    return solution["parsed"]


# Optional: With re-try to correct for failure to invoke tool
code_gen_chain = code_gen_chain_re_try | parse_output

# No re-try
code_gen_chain = code_gen_prompt_gemini | structured_llm_gemini | parse_output



from typing import List
from typing_extensions import TypedDict

# Graph State
class GraphState(TypedDict):
    error: str
    messages: List
    generation: str
    iterations: int

### Nodes
def validate_yaml(state: GraphState):
    """
    Validate YAML against ICL schema and decide next step.
    """

    print("---VALIDATING YAML---")

    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]
    yaml_code = code_solution.code  # Assuming YAML is returned as a string

    # Step 1: Validate YAML Syntax
    try:
        parsed_yaml = yaml.safe_load(yaml_code)  # Load YAML safely
    except yaml.YAMLError as e:
        print("---YAML SYNTAX ERROR---")
        error_message = f"Invalid YAML format: {e}"
        messages.append(("user", error_message))
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}

    # Step 2: Validate Against Schema
    try:
        validate(instance=parsed_yaml, schema=ICL_SCHEMA)
    except ValidationError as e:
        print("---SCHEMA VALIDATION FAILED---")
        error_message = f"Schema validation failed: {e.message}"
        messages.append(("user", error_message))
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}

    # If YAML is valid
    print("---YAML VALIDATION SUCCESSFUL---")
    return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"}

def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [
            (
                "user",
                "Fix the errors mentioned above and generate a corrected YAML file.",
            )
        ]

    # Solution
    code_solution = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.prefix}  \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

def decide_to_finish(state):
    """
    Determines whether to end the process or retry.
    """

    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations >= 7:  # Max 7 retries
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY YAML GENERATION---")
        return "generate"


        
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("generate", generate)
workflow.add_node("validate", validate_yaml)

# Define edges
workflow.add_edge(START, "generate")  # Start with code generation
workflow.add_edge("generate", "validate")  # Validate generated YAML
workflow.add_conditional_edges(
    "validate",
    decide_to_finish,
    {   
        "end": END,         # If valid, finish
        "generate": "generate",  # If invalid, retry
    },
)

# Compile the workflow
app = workflow.compile()

def process_question(question):
    result = app.invoke({"messages": [("user", question)], "iterations": 0, "error": ""})
    return result["generation"].code



# import os
# os._exit(0)