import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

MODELS = [
    {"id": "gpt-4o",                        "label": "GPT-4o"},
    {"id": "gpt-4o-mini",                   "label": "GPT-4o Mini"},
    {"id": "Meta-Llama-3.1-70B-Instruct",   "label": "Llama 3.1 70B"},
    {"id": "Meta-Llama-3.1-8B-Instruct",    "label": "Llama 3.1 8B"},
    {"id": "Mistral-large",                 "label": "Mistral Large"},
    {"id": "Mistral-small",                 "label": "Mistral Small"},
    {"id": "Phi-3.5-mini-instruct",         "label": "Phi-3.5 Mini"},
    {"id": "Cohere-command-r-plus",         "label": "Cohere Command R+"},
]

@app.route("/")
def index():
    return render_template("index.html", models=MODELS)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])
    model = data.get("model", "gpt-4o-mini")

    valid_ids = {m["id"] for m in MODELS}
    if model not in valid_ids:
        return jsonify({"error": "Invalid model selected."}), 400

    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    reply = response.choices[0].message.content
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
