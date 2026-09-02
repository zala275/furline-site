# Furline — Furniture Service Site

A small website where customers describe a furniture issue, a machine
learning model automatically sorts it into a category (Assembly Issue,
Damage/Defect, Delivery Problem, Repair Request, General Inquiry), and
the details get emailed straight to you.

Three files do the real work:
- `train_model.py` — trains the ML classifier (takes a few seconds)
- `app.py` — the website + the part that sends email
- `templates/index.html` — the page the customer sees

## Run it locally (about 10 minutes)

```bash
pip install -r requirements.txt
python train_model.py        # creates model/classifier.pkl
```

Set your email details as environment variables (see below), then:

```bash
python app.py
```

Open `http://localhost:5000` in your browser — the form is live.

## Setting up email (about 10 minutes)

The site uses Gmail's SMTP to send submissions to **ghanshyamliningwork@gmail.com**
(already set as the default — you don't need to configure this part).

You still need a sending email account and its app password — this is the
account the website sends *from*. You can use the same Gmail address or a
different one; it doesn't have to be ghanshyamliningwork@gmail.com itself.

1. Go to your Google Account → Security → 2-Step Verification (turn it
   on if it isn't already).
2. Go to myaccount.google.com/apppasswords and create an app password
   for "Mail".
3. Set these environment variables before running `python app.py`:

```bash
export SMTP_USERNAME="yoursendingaddress@gmail.com"
export SMTP_PASSWORD="the 16-character app password"
python app.py
```

Using a different email provider to send from? Change `SMTP_SERVER` and
`SMTP_PORT` in `app.py` to match (e.g. Outlook: `smtp.office365.com`,
port `587`).

## Putting it online (about 15 minutes)

Any free Python host works. **Render.com** is a straightforward option:

1. Push this folder to a GitHub repo.
2. On Render: New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt && python train_model.py`
4. Start command: `python app.py`
5. Add the three environment variables (`SMTP_USERNAME`, `SMTP_PASSWORD`,
   `OWNER_EMAIL`) in Render's dashboard.
6. Deploy — you'll get a public URL to share with customers.

## Improving the classifier later

`train_model.py` has a `TRAINING_DATA` list near the top — short example
sentences paired with a category. As real customers submit issues, copy
a few real (anonymized) examples into that list under the right category
and re-run `python train_model.py`. More real examples = better accuracy.
It doesn't need many — the current 40 examples already classify cleanly.
