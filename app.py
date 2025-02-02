from flask import Flask, request, jsonify
from flask_cors import CORS
from main_logic import process_question
import os

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
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT if available
    app.run(host="0.0.0.0", port=port, debug=True)
