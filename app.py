import os
import io
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from pypdf import PdfReader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

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

MAX_PDF_CHARS = 12000  # keep context manageable


@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["pdf"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    try:
        reader = PdfReader(io.BytesIO(file.read()))
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if not text:
            return jsonify({"error": "Could not extract text from this PDF. It may be scanned/image-based."}), 422

        truncated = len(text) > MAX_PDF_CHARS
        text = text[:MAX_PDF_CHARS]

        return jsonify({
            "text": text,
            "pages": len(reader.pages),
            "truncated": truncated,
            "filename": file.filename,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])
    model = data.get("model", "gpt-4o-mini")
    pdf_text = data.get("pdf_text", "")

    valid_ids = {m["id"] for m in MODELS}
    if model not in valid_ids:
        return jsonify({"error": "Invalid model selected."}), 400

    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    full_messages = []
    if pdf_text:
        full_messages.append({
            "role": "system",
            "content": (
                "The user has uploaded a PDF document. Use its content to answer questions.\n\n"
                f"--- PDF CONTENT ---\n{pdf_text}\n--- END PDF CONTENT ---"
            ),
        })
    full_messages.extend(messages)

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
    )

    reply = response.choices[0].message.content
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
