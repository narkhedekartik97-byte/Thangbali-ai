import os
import io
import json
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import OpenAI
from pypdf import PdfReader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please upload a PDF under 32 MB."}), 413

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

MODELS = [
    {"id": "gpt-4o",                       "label": "GPT-4o",          "tag": "Smart"},
    {"id": "gpt-4o-mini",                  "label": "GPT-4o Mini",     "tag": "Fast"},
    {"id": "Meta-Llama-3.1-70B-Instruct",  "label": "Llama 3.1 70B",  "tag": "Open"},
    {"id": "Meta-Llama-3.1-8B-Instruct",   "label": "Llama 3.1 8B",   "tag": "Lite"},
    {"id": "Mistral-large",                "label": "Mistral Large",   "tag": "EU"},
    {"id": "Mistral-small",                "label": "Mistral Small",   "tag": "Lite"},
    {"id": "Phi-3.5-mini-instruct",        "label": "Phi-3.5 Mini",   "tag": "Edge"},
    {"id": "Cohere-command-r-plus",        "label": "Command R+",      "tag": "RAG"},
]

MAX_PDF_CHARS = 12000


def build_messages(messages, pdf_text, system_prompt=None):
    full = []
    if system_prompt:
        full.append({"role": "system", "content": system_prompt})
    if pdf_text:
        full.append({
            "role": "system",
            "content": (
                "The user has uploaded a PDF. Use its content to answer questions accurately.\n\n"
                f"--- PDF CONTENT START ---\n{pdf_text}\n--- PDF CONTENT END ---"
            ),
        })
    full.extend(messages)
    return full


@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    messages     = data.get("messages", [])
    model        = data.get("model", "gpt-4o-mini")
    pdf_text     = data.get("pdf_text", "")
    system_prompt = data.get("system_prompt", "")

    valid_ids = {m["id"] for m in MODELS}
    if model not in valid_ids:
        def err():
            yield f"data: {json.dumps({'error': 'Invalid model.'})}\n\n"
        return Response(stream_with_context(err()), content_type="text/event-stream")

    if not messages:
        def err():
            yield f"data: {json.dumps({'error': 'No messages provided.'})}\n\n"
        return Response(stream_with_context(err()), content_type="text/event-stream")

    full_messages = build_messages(messages, pdf_text, system_prompt)

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=full_messages,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/chat", methods=["POST"])
def chat():
    """Non-streaming fallback."""
    data = request.get_json()
    messages      = data.get("messages", [])
    model         = data.get("model", "gpt-4o-mini")
    pdf_text      = data.get("pdf_text", "")
    system_prompt = data.get("system_prompt", "")

    valid_ids = {m["id"] for m in MODELS}
    if model not in valid_ids:
        return jsonify({"error": "Invalid model selected."}), 400
    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    full_messages = build_messages(messages, pdf_text, system_prompt)
    response = client.chat.completions.create(model=model, messages=full_messages)
    if response.choices:
        reply = response.choices[0].message.content
    else:
        reply = "No response generated."
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
