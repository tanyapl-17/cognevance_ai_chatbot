
import json
import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
INTENTS_FILE = BASE_DIR / "intents.json"
MODEL_FILE = BASE_DIR / "chatbot_model.pkl"


def train_chatbot():
    # Load the training examples
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []
    labels = []
    responses = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]

        # The fallback intent is used when confidence is low.
        if tag == "fallback":
            continue

        for pattern in intent["patterns"]:
            pattern = pattern.strip()

            if pattern:
                texts.append(pattern)
                labels.append(tag)

    if not texts:
        raise ValueError("No training patterns found in intents.json.")

    if len(set(labels)) < 2:
        raise ValueError("At least two different intent tags are required.")

    # Convert intent names into numeric labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Build the NLP classification pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                strip_accents="unicode",
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Train using all available examples
    model.fit(texts, encoded_labels)

    # Save the trained model and related information
    bundle = {
        "model": model,
        "encoder": encoder,
        "responses": responses,
        "threshold": 0.25
    }

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(bundle, file)

    print("\n" + "=" * 45)
    print("       NEXA AI MODEL TRAINED SUCCESSFULLY")
    print("=" * 45)
    print(f"Training examples: {len(texts)}")
    print(f"Intent categories: {len(set(labels))}")
    print(f"Model saved to: {MODEL_FILE}")
    print("\nAvailable intents:")

    for tag in encoder.classes_:
        print(f" - {tag}")

    print("\nYou can now run: python -m streamlit run app.py")


if __name__ == "__main__":
    train_chatbot()
