from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import logging
import webbrowser

from pyspark.ml.tuning import CrossValidatorModel

from lyrics_classifier import LogisticRegressionPipeline

# Get the current working directory (project folder)
project_dir = os.path.dirname(os.path.abspath(__file__))

# Set the relative path to the Python executable in your virtual environment
venv_python = os.path.join(project_dir, "venv", "bin", "python")

# Set the PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON environment variables
os.environ["PYSPARK_PYTHON"] = venv_python
os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

pipeline: LogisticRegressionPipeline
model: CrossValidatorModel

DATASET_PATH = os.path.abspath("data/Merged_dataset.csv")
MODEL_DIR_PATH = os.path.abspath("model_combined/")

# DATASET_PATH = os.path.abspath("data/Mendeley_dataset.csv")
# MODEL_DIR_PATH = os.path.abspath("model_mendeley/")


def load_model():
    global pipeline
    global model

    if not (
        os.path.exists(MODEL_DIR_PATH)
        and os.path.isdir(MODEL_DIR_PATH)
        and len(os.listdir(MODEL_DIR_PATH)) > 0
    ):
        os.makedirs(MODEL_DIR_PATH, exist_ok=True)

        pipeline = LogisticRegressionPipeline()

        print("Training the model...")

        model = pipeline.train_and_test(
            dataset_path=DATASET_PATH,
            train_ratio=0.8,
            store_model_on=MODEL_DIR_PATH,
            print_statistics=True,
        )

        model.save(MODEL_DIR_PATH)
        print("Model trained and saved successfully.")

    else:
        pipeline = LogisticRegressionPipeline()

        print("Loading the pre trained model...")

        model = CrossValidatorModel.load(MODEL_DIR_PATH)

    return True


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


# Function to classify lyrics
def predict_genre(lyrics):
    if not lyrics.strip():
        return None

    threshold = 0.35
    prediction, probabilities = pipeline.predict_one(
        unknown_lyrics=lyrics,
        threshold=threshold,
        model=model,
    )

    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")

    return {"predicted_genre": prediction, "probabilities": probabilities}


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
    loaded = load_model()
    if loaded:
        # Create templates directory if it doesn't exist
        os.makedirs("templates", exist_ok=True)

        # Copy index.html to templates directory for Flask to find it
        if not os.path.exists("templates/index.html"):
            with open("index.html", "r") as f_src:
                with open("templates/index.html", "w") as f_dst:
                    f_dst.write(f_src.read())

        webbrowser.open("http://10.10.43.93:5000/")
        # Run the Flask app
        app.run(debug=True, host="0.0.0.0", port=5000)

    else:
        logger.error("Failed to load model. Exiting...")
