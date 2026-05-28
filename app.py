import os
import io
import json
import uuid
import time
import sqlite3
import requests as http_req
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, g
from openai import OpenAI
from pypdf import PdfReader

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")
STATIC_GEN = os.path.join(os.path.dirname(__file__), "static", "generated")
os.makedirs(STATIC_GEN, exist_ok=True)


# ── Database ─────────────────────────────────────────────────────────────────

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                pdf_name   TEXT NOT NULL DEFAULT '',
                pdf_text   TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        db.commit()


init_db()


# ── AI Clients ────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please upload a PDF under 32 MB."}), 413


chat_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

_openai_key = os.environ.get("OPENAI_API_KEY", "")
image_client = OpenAI(api_key=_openai_key) if _openai_key else None

MODELS = [
    {"id": "gpt-4o",                       "label": "GPT-4o",         "tag": "Smart"},
    {"id": "gpt-4o-mini",                  "label": "GPT-4o Mini",    "tag": "Fast"},
    {"id": "Meta-Llama-3.1-70B-Instruct",  "label": "Llama 3.1 70B", "tag": "Open"},
    {"id": "Meta-Llama-3.1-8B-Instruct",   "label": "Llama 3.1 8B",  "tag": "Lite"},
    {"id": "Mistral-large",                "label": "Mistral Large",  "tag": "EU"},
    {"id": "Mistral-small",                "label": "Mistral Small",  "tag": "Lite"},
    {"id": "Phi-3.5-mini-instruct",        "label": "Phi-3.5 Mini",  "tag": "Edge"},
    {"id": "Cohere-command-r-plus",        "label": "Command R+",     "tag": "RAG"},
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


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


# ── Sessions API ──────────────────────────────────────────────────────────────

@app.route("/sessions", methods=["GET"])
def list_sessions():
    db = get_db()
    rows = db.execute(
        "SELECT id, title, pdf_name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/sessions", methods=["POST"])
def create_session():
    data  = request.get_json(silent=True) or {}
    title = (data.get("title") or "New Chat")[:80]
    now   = time.time()
    sid   = str(uuid.uuid4())
    db    = get_db()
    db.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    db.commit()
    return jsonify({"id": sid, "title": title, "created_at": now, "updated_at": now}), 201


@app.route("/sessions/<sid>", methods=["PATCH"])
def update_session(sid):
    data = request.get_json(silent=True) or {}
    db   = get_db()
    fields, vals = [], []
    if "title" in data:
        fields.append("title = ?"); vals.append(data["title"][:80])
    if "pdf_name" in data:
        fields.append("pdf_name = ?"); vals.append(data["pdf_name"])
    if "pdf_text" in data:
        fields.append("pdf_text = ?"); vals.append(data["pdf_text"])
    if not fields:
        return jsonify({"error": "Nothing to update"}), 400
    fields.append("updated_at = ?"); vals.append(time.time())
    vals.append(sid)
    db.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?", vals)
    db.commit()
    return jsonify({"ok": True})


@app.route("/sessions/<sid>", methods=["DELETE"])
def delete_session(sid):
    db = get_db()
    db.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    db.commit()
    return jsonify({"ok": True})


# ── Messages API ──────────────────────────────────────────────────────────────

@app.route("/sessions/<sid>/messages", methods=["GET"])
def get_messages(sid):
    db   = get_db()
    sess = db.execute("SELECT id, pdf_name, pdf_text FROM sessions WHERE id = ?", (sid,)).fetchone()
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    msgs = db.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at",
        (sid,),
    ).fetchall()
    return jsonify({
        "session":  dict(sess),
        "messages": [dict(m) for m in msgs],
    })


@app.route("/sessions/<sid>/messages", methods=["POST"])
def add_messages(sid):
    db   = get_db()
    sess = db.execute("SELECT id FROM sessions WHERE id = ?", (sid,)).fetchone()
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json(silent=True) or {}
    msgs = data.get("messages", [])
    if not msgs:
        return jsonify({"error": "No messages provided"}), 400
    now = time.time()
    db.executemany(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        [(sid, m["role"], m["content"], now + i * 0.001) for i, m in enumerate(msgs)],
    )
    db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
    db.commit()
    return jsonify({"ok": True, "count": len(msgs)}), 201


# ── Image Generation ──────────────────────────────────────────────────────────

@app.route("/generate/image", methods=["POST"])
def generate_image():
    if not image_client:
        return jsonify({"error": "OPENAI_API_KEY is not configured."}), 500
    data   = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    size    = data.get("size", "1024x1024")
    quality = data.get("quality", "standard")
    style   = data.get("style", "vivid")
    if size not in {"1024x1024", "1792x1024", "1024x1792"}:
        size = "1024x1024"

    try:
        resp = image_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            response_format="b64_json",
            n=1,
        )
        b64     = resp.data[0].b64_json
        revised = resp.data[0].revised_prompt or prompt
        return jsonify({"image": f"data:image/png;base64,{b64}", "revised_prompt": revised})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Video Generation ──────────────────────────────────────────────────────────

@app.route("/generate/video", methods=["POST"])
def generate_video():
    data   = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    HF_MODEL = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"
    hf_token = os.environ.get("HF_TOKEN", "")
    headers  = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    last_error = "Unknown error"
    for attempt in range(6):
        try:
            resp = http_req.post(
                HF_MODEL,
                headers=headers,
                json={"inputs": prompt},
                timeout=120,
            )
            if resp.status_code == 200:
                filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
                filepath = os.path.join(STATIC_GEN, filename)
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                return jsonify({"video": f"/static/generated/{filename}"})
            elif resp.status_code == 503:
                try:
                    wait = float(resp.json().get("estimated_time", 20))
                except Exception:
                    wait = 20
                time.sleep(min(wait, 30))
            else:
                try:
                    last_error = resp.json().get("error", resp.text[:300])
                except Exception:
                    last_error = resp.text[:300]
                break
        except Exception as e:
            last_error = str(e)
            if attempt < 5:
                time.sleep(5)

    return jsonify({"error": f"Video generation failed: {last_error}"}), 500


# ── Chat (streaming) ──────────────────────────────────────────────────────────

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data          = request.get_json()
    messages      = data.get("messages", [])
    model         = data.get("model", "gpt-4o-mini")
    pdf_text      = data.get("pdf_text", "")
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
            stream = chat_client.chat.completions.create(
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


# ── Chat (non-streaming fallback) ─────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data          = request.get_json()
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
    response = chat_client.chat.completions.create(model=model, messages=full_messages)
    if response.choices:
        reply = response.choices[0].message.content
    else:
        reply = "No response generated."
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
