import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "gemini-3.1-flash-lite"

app = Flask(__name__, template_folder="templates")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

config_path = BASE_DIR / "chatbot_config"
SYSTEM_PROMPT = config_path.read_text(encoding="utf-8").strip()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    if client is None:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a study question."}), 400

    prompt = f"{SYSTEM_PROMPT}\n\nStudent question:\n{message}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        answer = getattr(response, "text", None) or "I couldn't generate a response."
        return jsonify({"answer": answer})
    except Exception:
        return jsonify(
            {"error": "Unable to reach Gemini right now. Please try again."}
        ), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
