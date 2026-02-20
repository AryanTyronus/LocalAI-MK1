from flask import Flask, request, jsonify, render_template
from services.ai_service import AIService

app = Flask(__name__)
ai_service = AIService()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()

    if not data:
        return jsonify({"response": "Invalid request format."}), 400

    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    try:
        reply = ai_service.ask(user_message)

        # Standardized response key
        return jsonify({
            "response": reply
        })

    except Exception as e:
        print("Backend Error:", str(e))
        return jsonify({
            "response": "Something went wrong while generating a response."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)