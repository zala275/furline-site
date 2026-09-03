"""
Furniture business service-request site.

Flow:
  1. Customer fills out the form (name, contact, issue description).
  2. The ML model (trained by train_model.py) classifies the issue text
     into a category and flags urgency.
  3. All details + the ML prediction are emailed to the business owner.

Setup:
  1. pip install -r requirements.txt
  2. python train_model.py          (creates model/classifier.pkl)
  3. Set the environment variables below (see README.md for how to get
     a Gmail "app password", or swap in any other SMTP provider).
  4. python app.py
"""

import os
import pickle
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "model" / "classifier.pkl"
_model = None

URGENT_KEYWORDS = [
    "broken", "damaged", "crack", "unsafe", "collapsed", "injury", "hurt",
    "urgent", "asap", "immediately", "still hasn't arrived", "no one showed up",
    "defective", "stuck", "torn", "chip", "dent",
]

# --- Email settings (set these as environment variables before running) --
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")  # your sending email address
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")  # app password, not your normal password
OWNER_EMAIL = os.environ.get("OWNER_EMAIL", "ghanshyamliningwork@gmail.com")  # where submissions are sent


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


def send_email(name, furniture_type, phone, email, issue_text, prediction):
    if not (SMTP_USERNAME and SMTP_PASSWORD):
        raise RuntimeError(
            "Email is not configured. Set SMTP_USERNAME and SMTP_PASSWORD "
            "environment variables (see README.md)."
        )

    urgency_line = "HIGH PRIORITY" if prediction["urgent"] else "Normal"

    body = f"""New service request from your website

Customer name: {name}
Furniture type: {furniture_type}
Phone: {phone}
Email: {email}

Predicted category: {prediction['category']} ({prediction['confidence']}% confidence)
Urgency: {urgency_line}

Issue description:
{issue_text}
"""

    msg = MIMEText(body)
    msg["Subject"] = f"Furline: [{prediction['category']}] New request from {name}"
    msg["From"] = SMTP_USERNAME
    msg["To"] = OWNER_EMAIL

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, [OWNER_EMAIL], msg.as_string())


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
        send_email(name, furniture_type, phone, email, issue_text, prediction)
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
