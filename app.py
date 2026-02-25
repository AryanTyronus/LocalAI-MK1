from flask import Flask, request, jsonify, render_template
from services.ai_service import AIService
import os
import socket

app = Flask(__name__)
from core.dependency_container import DependencyContainer

_container = None
_ai_service = None

def get_ai_service():
    global _container, _ai_service
    if _ai_service is None:
        try:
            _container = DependencyContainer()
            _ai_service = _container.get_ai_service()
        except Exception as e:
            # Log and print full stack trace
            import traceback
            print(f"\n=== CRITICAL: AI SERVICE INITIALIZATION FAILED ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            print(f"===================================================\n")
            raise
    return _ai_service

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()

    if not data:
        return jsonify({"response": "Invalid request format."}), 400

    user_message = data.get("message", "").strip()
    mode = data.get("mode", "chat")  # default to 'chat'

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    try:
        svc = None
        try:
            svc = get_ai_service()
        except Exception as e:
            # Model initialization failed
            import traceback
            traceback.print_exc()
            print(f"\n=== AI SERVICE INIT ERROR ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"=============================\n")
            
            # Import DEBUG config
            from core.config import DEBUG_MODE
            
            if DEBUG_MODE:
                return jsonify({
                    "response": f"Service initialization failed: {type(e).__name__}: {str(e)}",
                    "error": str(e),
                    "error_type": type(e).__name__
                }), 503
            else:
                return jsonify({"response": "Model initialization failed. Please try again later."}), 503

        reply = svc.ask(user_message, mode=mode)
        return jsonify({"response": reply})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n=== CHAT ENDPOINT ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"User message: {user_message}")
        print(f"Mode: {mode}")
        print(f"===========================\n")
        
        # Import DEBUG config
        from core.config import DEBUG_MODE
        
        if DEBUG_MODE:
            return jsonify({
                "response": f"Backend error: {type(e).__name__}: {str(e)}",
                "error": str(e),
                "error_type": type(e).__name__
            }), 500
        else:
            return jsonify({"response": "Something went wrong while generating a response."}), 500


def _find_free_port(preferred: int) -> int:
    """Return preferred if available, otherwise scan next ports up to +50."""
    for port in range(preferred, preferred + 51):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    # Fallback to 0 -> let OS pick
    return 0


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

    # Optionally preload model (will instantiate DependencyContainer)
    if args.preload_model:
        try:
            print("Preloading model and services...")
            get_ai_service()
            print("Preload complete.")
        except Exception as e:
            print(f"Preload failed: {e}")

    # Allow dev mode env to be set externally (e.g., LOCALAI_DEV_MODE=1)
    app.run(host="0.0.0.0", port=chosen, debug=False)