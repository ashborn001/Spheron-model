from flask import Flask, request, jsonify
from flask_cors import CORS
from main_logic import process_question

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

@app.route("/")
def home():
    return jsonify({"message": "Flask API for Spheron YAML generation is running!"})

@app.route("/generate_yaml", methods=["POST"])
def generate_yaml():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        yaml_output = process_question(question)
        return jsonify({"yaml": yaml_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
