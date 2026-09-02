"""
Trains a small text classifier that reads a customer's furniture-issue
description and predicts which category it belongs to, plus how urgent
it looks. Run this once (python train_model.py) to produce model/classifier.pkl.

You can improve accuracy over time by adding more example sentences below
to match the kinds of issues your real customers write in.
"""

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- Training data -----------------------------------------------------
# Each tuple is (example customer text, category label).
# Add real examples from your own customers here over time to improve it.

TRAINING_DATA = [
    # Assembly Issue
    ("I can't figure out how to assemble the bed frame, some screws are missing", "Assembly Issue"),
    ("The instructions for the wardrobe don't match the parts we received", "Assembly Issue"),
    ("We are stuck putting together the dining table, one leg won't attach", "Assembly Issue"),
    ("Missing hardware bag for the bookshelf assembly", "Assembly Issue"),
    ("The drawer slides don't line up with the holes during assembly", "Assembly Issue"),
    ("Need help assembling the sofa frame, the manual is unclear", "Assembly Issue"),
    ("Bolts provided are the wrong size for the shelf unit", "Assembly Issue"),
    ("Can someone help me put together my new desk, I'm missing a panel", "Assembly Issue"),

    # Damage/Defect
    ("The wooden table arrived with a large crack down the middle", "Damage/Defect"),
    ("There is a deep scratch across the top of the dresser", "Damage/Defect"),
    ("One of the chair legs is broken and wobbly", "Damage/Defect"),
    ("The fabric on the sofa is torn near the armrest", "Damage/Defect"),
    ("The finish is peeling off the cabinet after two weeks", "Damage/Defect"),
    ("My couch cushion foam is already sagging and looks defective", "Damage/Defect"),
    ("There's a stain on the leather recliner that won't come off, seems like a factory defect", "Damage/Defect"),
    ("The glass tabletop has a chip in the corner", "Damage/Defect"),

    # Delivery Problem
    ("My order still hasn't arrived and it's been three weeks", "Delivery Problem"),
    ("The delivery team damaged my doorway bringing in the couch", "Delivery Problem"),
    ("I received the wrong color for my ordered dining set", "Delivery Problem"),
    ("Only half of my furniture set was delivered", "Delivery Problem"),
    ("The tracking says delivered but I never received the package", "Delivery Problem"),
    ("Delivery was scheduled but no one showed up", "Delivery Problem"),
    ("The box was crushed during shipping and the item inside is dented", "Delivery Problem"),
    ("I need to reschedule my delivery date for the bed", "Delivery Problem"),

    # Repair Request
    ("My recliner mechanism is stuck and won't recline anymore", "Repair Request"),
    ("The hinge on my cabinet door has come loose", "Repair Request"),
    ("One of the dining chairs is squeaking loudly, needs a fix", "Repair Request"),
    ("Can you send a technician to fix the sofa bed mechanism", "Repair Request"),
    ("The drawer on my nightstand won't open smoothly anymore", "Repair Request"),
    ("Table wobbles because a leg has come loose over time", "Repair Request"),
    ("Zipper on the couch cushion cover is broken", "Repair Request"),
    ("Need a repair visit for a wobbly office chair", "Repair Request"),

    # General Inquiry
    ("Do you offer a warranty on your dining sets", "General Inquiry"),
    ("What is the estimated delivery time for a new order", "General Inquiry"),
    ("Can I get a quote for custom furniture", "General Inquiry"),
    ("What materials are used in your outdoor furniture line", "General Inquiry"),
    ("Do you offer white glove delivery service", "General Inquiry"),
    ("I want to know your store hours and location", "General Inquiry"),
    ("Can I return an item if I don't like the color", "General Inquiry"),
    ("Do you have a showroom I can visit before ordering", "General Inquiry"),
]

URGENT_KEYWORDS = [
    "broken", "damaged", "crack", "unsafe", "collapsed", "injury", "hurt",
    "urgent", "asap", "immediately", "still hasn't arrived", "no one showed up",
    "defective", "won't", "stuck", "torn", "chip", "dent",
]


def build_pipeline():
    texts = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def main():
    pipeline = build_pipeline()

    out_dir = Path(__file__).parent / "model"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "classifier.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model trained on {len(TRAINING_DATA)} examples and saved to {out_dir / 'classifier.pkl'}")

    # quick sanity check
    samples = [
        "The chair I bought is wobbly and one leg is loose",
        "My delivery never arrived and it's been two weeks",
        "Do you have a warranty on sofas",
    ]
    for s in samples:
        pred = pipeline.predict([s])[0]
        conf = pipeline.predict_proba([s]).max()
        print(f"  '{s}' -> {pred} ({conf:.0%} confidence)")


if __name__ == "__main__":
    main()
