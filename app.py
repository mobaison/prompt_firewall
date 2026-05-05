"""
app.py - Main Flask Application
================================
Flow for every chat message:
  1. Receive user message
  2. Run through PromptFirewall (5 layers)
       → blocked?  return rejection / redirect response immediately
       → allowed?  continue
  3. Pass clean message to RAG pipeline
  4. Return grounded answer
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os, uuid
from dotenv import load_dotenv

from rag.pipeline      import RAGPipeline
from firewall.firewall import PromptFirewall

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "hospital-chatbot-secret-2024")
CORS(app)

# ── Initialize both systems at startup ────────────────────────────
print("🔄 Initializing RAG Pipeline...")
rag = RAGPipeline()
print("✅ RAG Pipeline ready!\n")

print("🛡️  Initializing Prompt Firewall...")
firewall = PromptFirewall()
print("✅ Firewall ready!\n")


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    session_id   = session.get("session_id", "default")

    # ══════════════════════════════════════════════
    # FIREWALL — runs before anything else
    # ══════════════════════════════════════════════
    fw = firewall.check(user_message, session_id=session_id)

    if not fw.allowed:
        # Return the firewall's message directly — RAG never runs
        response_body = {
            "response"  : fw.message,
            "sources"   : [],
            "blocked"   : True,
            "block_layer": fw.layer,
            "action"    : fw.action,     # "block" or "redirect"
        }
        # For crisis redirects use 200 so the UI renders the message normally
        return jsonify(response_body), 200

    # ══════════════════════════════════════════════
    # RAG PIPELINE — only runs if firewall passes
    # ══════════════════════════════════════════════
    try:
        result = rag.query(user_message, session_id=session_id)

        response_body = {
            "response"  : result["answer"],
            "sources"   : result.get("sources", []),
            "blocked"   : False,
            "confidence": result.get("confidence", "high"),
        }

        # Attach soft rate-limit warning if approaching limits
        if fw.warning:
            response_body["warning"] = fw.warning

        return jsonify(response_body)

    except Exception as e:
        print(f"❌ RAG error: {e}")
        return jsonify({
            "response": (
                "I'm having trouble right now. "
                "Please call City General Hospital at +1-800-HOSPITAL."
            ),
            "sources" : [],
            "blocked" : False,
        }), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status"    : "ok",
        "rag_ready" : rag.is_ready(),
        "firewall"  : "active",
        "layers"    : 5,
    })


@app.route("/api/firewall/stats")
def firewall_stats():
    """Return rate-limit stats for current session (debug endpoint)."""
    session_id = session.get("session_id", "default")
    return jsonify(firewall.stats(session_id))  


@app.route("/api/clear", methods=["POST"])
def clear_history():
    session_id = session.get("session_id", "default")
    rag.clear_history(session_id)
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)