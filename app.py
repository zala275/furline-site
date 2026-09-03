"""
Furniture business service-request site.

Flow:
  1. Customer fills out the form (name, furniture type, issue, phone, email).
  2. The ML model (trained by train_model.py) classifies the issue text
     into a category and flags urgency.
  3. All details + the ML prediction are sent to Formspree, which forwards
     them by email to the business owner. No email login/password needed.

Setup:
  1. pip install -r requirements.txt
  2. python train_model.py          (creates model/classifier.pkl)
  3. python app.py
"""

import os
import pickle
from pathlib import Path

import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "model" / "classifier.pkl"
_model = None

URGENT_KEYWORDS = [
    "broken", "damaged", "crack", "unsafe", "collapsed", "injury", "hurt",
    "urgent", "asap", "immediately", "still hasn't arrived", "no one showed up",
    "defective", "stuck", "torn", "chip", "dent",
]

# Formspree endpoint that forwards submissions to ghanshyamliningwork@gmail.com
FORMSPREE_URL = "https://formspree.io/f/maeyjkgn"


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                "Model file not found. Run 'python train_model.py' first."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def classify_issue(text: str):
    model = get_model()
    category = model.predict([text])[0]
    confidence = float(model.predict_proba([text]).max())

    text_lower = text.lower()
    is_urgent = any(kw in text_lower for kw in URGENT_KEYWORDS)

    return {
        "category": category,
        "confidence": round(confidence * 100, 1),
        "urgent": is_urgent,
    }


def send_via_formspree(name, furniture_type, phone, email, issue_text, prediction):
    urgency_line = "HIGH PRIORITY" if prediction["urgent"] else "Normal"

    payload = {
        "name": name,
        "furniture_type": furniture_type,
        "phone": phone,
        "email": email,
        "issue": issue_text,
        "predicted_category": prediction["category"],
        "confidence": f"{prediction['confidence']}%",
        "urgency": urgency_line,
        "_subject": f"Furline: [{prediction['category']}] New request from {name}",
    }

    response = requests.post(
        FORMSPREE_URL,
        data=payload,
        headers={"Accept": "application/json"},
        timeout=15,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Formspree error: {response.status_code} {response.text}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    name = request.form.get("name", "").strip()
    furniture_type = request.form.get("furniture_type", "").strip()
    phone = request.form.get("phone", "").strip()
    email = request.form.get("email", "").strip()
    issue_text = request.form.get("issue", "").strip()

    if not name or not furniture_type or not phone or not email or not issue_text:
        return jsonify({"ok": False, "error": "Please fill out all fields."}), 400

    prediction = classify_issue(issue_text)

    try:
        send_via_formspree(name, furniture_type, phone, email, issue_text, prediction)
    except RuntimeError as e:
        print(f"Email not sent: {e}")
        return jsonify({
            "ok": True,
            "email_sent": False,
            "category": prediction["category"],
        })

    return jsonify({
        "ok": True,
        "email_sent": True,
        "category": prediction["category"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
