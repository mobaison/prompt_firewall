"""
app.py - Main Flask Application Entry Point
==========================================
This is the core of the chatbot. It:
- Initializes Flask and all routes
- Loads the RAG pipeline on startup
- Handles chat API requests from the frontend
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import uuid
from dotenv import load_dotenv
from rag.pipeline import RAGPipeline

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "hospital-chatbot-secret-2024")
CORS(app)  # Allow cross-origin requests (useful during development)

# -----------------------------------------------
# Initialize RAG Pipeline once at startup
# -----------------------------------------------
print("🔄 Initializing RAG Pipeline...")
rag = RAGPipeline()
print("✅ RAG Pipeline ready!")


# -----------------------------------------------
# Routes
# -----------------------------------------------

@app.route("/")
def index():
    """Serve the main chatbot UI page."""
    # Give each browser tab a unique session ID for conversation history
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body: { "message": "user question here" }
    Returns: { "response": "bot answer", "sources": [...] }
    
    This is the main endpoint the frontend JavaScript calls.
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", "default")

    try:
        # Run through RAG pipeline: retrieve context → generate answer
        result = rag.query(user_message, session_id=session_id)
        return jsonify({
            "response": result["answer"],
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", "high")
        })
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({
            "response": "I'm sorry, I encountered an error. Please try again.",
            "sources": [],
            "confidence": "error"
        }), 500


@app.route("/api/health")
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "rag_ready": rag.is_ready()})


@app.route("/api/clear", methods=["POST"])
def clear_history():
    """Clear conversation history for the current session."""
    session_id = session.get("session_id", "default")
    rag.clear_history(session_id)
    return jsonify({"status": "cleared"})


# -----------------------------------------------
# Run
# -----------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5009)