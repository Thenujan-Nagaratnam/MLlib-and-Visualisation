from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model and vectorizer paths
MODEL_PATH = "model/genre_classifier.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Initialize model and vectorizer as None
model = None
vectorizer = None

# Dictionary mapping of genres (should match your model's classes)
# Update this based on your actual model's classes
GENRES = [
    "pop",
    "rock",
    "rap",
    "country",
    "r&b",
    "electronic",
    "jazz",
    "blues",
    "folk",
    "metal",
]


def load_model():
    """Load the pre-trained model and vectorizer"""
    global model, vectorizer

    try:
        logger.info("Loading model and vectorizer...")

        # Check if model exists, if not create dummy model for demonstration
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            logger.warning(
                "Model or vectorizer not found. Creating dummy model for demonstration."
            )
            create_dummy_model()

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        logger.info("Model and vectorizer loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


def create_dummy_model():
    """Create a simple dummy model and vectorizer for demonstration"""
    from sklearn.ensemble import RandomForestClassifier

    # Create a simple dummy vectorizer
    dummy_vectorizer = TfidfVectorizer(max_features=1000)
    # Fit on some example text
    dummy_vectorizer.fit(
        [
            "this is an example lyric for pop music",
            "rock music has guitars and drums",
            "rap music has rhymes and beats",
            "country music tells stories about life",
        ]
    )

    # Create a simple dummy model
    dummy_model = RandomForestClassifier(n_estimators=10)
    # Fit with dummy data
    dummy_X = dummy_vectorizer.transform(["example"] * 10)
    dummy_y = np.random.choice(GENRES, size=10)
    dummy_model.fit(dummy_X, dummy_y)

    # Save the dummy model and vectorizer
    joblib.dump(dummy_model, MODEL_PATH)
    joblib.dump(dummy_vectorizer, VECTORIZER_PATH)

    logger.info("Dummy model created and saved.")


def predict_genre(lyrics):
    """Predict genre from lyrics using the loaded model"""
    try:
        # Preprocess lyrics (get TF-IDF features)
        X = vectorizer.transform([lyrics])

        # Get prediction probabilities
        probabilities = model.predict_proba(X)[0]

        # Create dictionary of genre:probability
        genre_probs = {GENRES[i]: float(prob) for i, prob in enumerate(probabilities)}

        # Get the predicted genre (highest probability)
        predicted_genre = GENRES[np.argmax(probabilities)]

        return {"predicted_genre": predicted_genre, "probabilities": genre_probs}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Get lyrics from form
            lyrics = request.form.get("lyrics", "")

            if not lyrics:
                return jsonify({"error": "No lyrics provided"}), 400

            # Make prediction
            result = predict_genre(lyrics)

            if "error" in result:
                return jsonify({"error": result["error"]}), 500

            return jsonify(result)

        except Exception as e:
            logger.error(f"Error in prediction route: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    # Load model on startup
    if load_model():
        # Create templates directory if it doesn't exist
        os.makedirs("templates", exist_ok=True)

        # Copy index.html to templates directory for Flask to find it
        if not os.path.exists("templates/index.html"):
            with open("index.html", "r") as f_src:
                with open("templates/index.html", "w") as f_dst:
                    f_dst.write(f_src.read())

        # Run the Flask app
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        logger.error("Failed to load model. Exiting...")
