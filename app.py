"""
Flask application - HTTP layer with ZERO business logic.

This module contains ONLY:
- Request parsing
- Response formatting
- Error handling

NO memory logic, NO prompt building, NO model calls, NO config logic.
All business logic is in AIService.
"""

from flask import Flask, request, jsonify, render_template, Response
import os
import socket

from core.orchestrator import AppOrchestrator
from core.config import DEBUG_MODE, Config
from core.logger import logger

app = Flask(__name__)

# Global orchestrator (lazy initialization)
_orchestrator = AppOrchestrator()


def get_ai_service():
    """
    Get the AIService instance (lazy initialization).
    
    Returns:
        AIService instance
    """
    try:
        return _orchestrator.get_ai_service()
    except Exception as e:
        import traceback
        logger.error(f"AI SERVICE INITIALIZATION FAILED: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise


# ===================
# Routes
# ===================

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html", projects=Config().ui_projects)


@app.route("/chat", methods=["POST"])
def chat_api():
    """
    Chat endpoint - thin controller with ZERO business logic.
    
    Expected JSON payload:
        {
            "message": "user message",
            "mode": "chat"  // optional, defaults to "chat"
        }
    
    Returns:
        JSON response with "response" field
    """
    # 1. Parse request
    data = request.get_json(silent=True)
    
    if not data:
        return jsonify({"response": "Invalid request format."}), 400
    
    user_message = data.get("message", "").strip()
    mode = data.get("mode", "chat")
    options = data.get("options", {})
    
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    
    # 2. Call AIService (all business logic is here)
    try:
        svc = get_ai_service()
        
        # Use the new generate_response method
        reply = svc.generate_response(user_message, mode=mode, options=options)
        turn_meta = svc.get_last_turn_meta() if hasattr(svc, "get_last_turn_meta") else {}
        
        return jsonify({
            "response": reply,
            "memory_updated": bool(turn_meta.get("memory_updated", False))
        })
    
    except Exception as e:
        import traceback
        logger.error(f"Chat endpoint error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        
        if DEBUG_MODE:
            return jsonify({
                "response": f"Backend error: {type(e).__name__}: {str(e)}",
                "error": str(e),
                "error_type": type(e).__name__
            }), 500
        else:
            return jsonify({"response": "Something went wrong while generating a response."}), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    
    Expected JSON payload:
        {
            "message": "user message",
            "mode": "chat"  // optional, defaults to "chat"
        }
    
    Returns:
        Server-Sent Events stream with "data" field
    """
    # 1. Parse request
    data = request.get_json(silent=True)
    
    if not data:
        return jsonify({"response": "Invalid request format."}), 400
    
    user_message = data.get("message", "").strip()
    mode = data.get("mode", "chat")
    options = data.get("options", {})
    
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    
    # 2. Stream response using SSE
    def generate():
        try:
            svc = get_ai_service()
            
            for chunk in svc.generate_stream(user_message, mode=mode, options=options):
                # Send chunk as SSE data
                yield f"data: {chunk}\n\n"
                
        except Exception as e:
            import traceback
            logger.error(f"Streaming endpoint error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            
            error_msg = str(e) if DEBUG_MODE else "Streaming error occurred."
            yield f"data: __ERROR__:{error_msg}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


# ===================
# Utility Functions
# ===================

def _find_free_port(preferred: int) -> int:
    """
    Find a free port starting from preferred.
    
    Args:
        preferred: Preferred port number
        
    Returns:
        Available port number
    """
    for port in range(preferred, preferred + 51):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return 0


# ===================
# Main Entry Point
# ===================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--preload-model", action="store_true", help="Preload model at startup")
    args, _ = parser.parse_known_args()
    
    preferred = int(os.getenv('PORT', '8000'))
    chosen = _find_free_port(preferred)
    
    if chosen != preferred:
        print(f"Port {preferred} unavailable — starting on {chosen} instead.")
    else:
        print(f"Starting server on port {chosen}")
    
    # Optionally preload model
    if args.preload_model:
        try:
            print("Preloading model and services...")
            _orchestrator.warm_model()
            get_ai_service()
            print("Preload complete.")
        except Exception as e:
            print(f"Preload failed: {e}")
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=chosen, debug=False)
