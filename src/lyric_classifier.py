import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import os
import webbrowser

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.ml.tuning import CrossValidatorModel
from src.lyrics.services.pipelines.lr_pipeline import LogisticRegressionPipeline


DATASET_PATH = os.path.abspath("data/merged-dataset.csv")

MODEL_DIR_PATH = os.path.abspath("model/")

pipeline: LogisticRegressionPipeline = None
model: CrossValidatorModel = None
classifier = None


def load_model():
    if not (
        os.path.exists(MODEL_DIR_PATH)
        and os.path.isdir(MODEL_DIR_PATH)
        and len(os.listdir(MODEL_DIR_PATH)) > 0
    ):
        os.makedirs(MODEL_DIR_PATH, exist_ok=True)

        pipeline = LogisticRegressionPipeline()

        model = pipeline.train_and_test(
            dataset_path=DATASET_PATH,
            train_ratio=0.8,
            store_model_on=MODEL_DIR_PATH,
            print_statistics=True,
        )
    else:
        pipeline = LogisticRegressionPipeline()
        model = CrossValidatorModel.load(MODEL_DIR_PATH)

    global classifier
    classifier = model
    return pipeline, model


def on_shutdown():
    pipeline.stop()


# Set page configuration
st.set_page_config(
    page_title="Lyric Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6C5CE7;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #a29bfe;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .stTextArea label {
        font-size: 1.2rem;
        color: #6C5CE7;
        font-weight: 500;
    }
    .genre-header {
        font-size: 1.8rem;
        color: #6C5CE7;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 600;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #6C5CE7;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
    }
    .stButton button:hover {
        background-color: #5649c0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Define the genre labels
GENRES = [
    "Pop",
    "Rock",
    "Hip-Hop",
    "R&B",
    "Country",
    "Electronic",
    "Jazz",
    "Classical",
    "Reggae",
    "Metal",
]


# Function to classify lyrics
def classify_lyrics(lyrics, classifier):
    if not lyrics.strip():
        return None

    # For demonstration purposes, we'll map the model's general classification to music genres
    genre_hypotheses = [f"This is {genre} music lyrics" for genre in GENRES]

    with st.spinner("Analyzing lyrics..."):
        time.sleep(0.5)  # Simulate processing time

        # # Generate predictions for each genre hypothesis
        # prediction = pipeline.predict_one(
        #     unknown_lyrics=lyrics,
        #     threshold=0.35,
        #     model=classifier,
        # )
        for hypothesis in genre_hypotheses:
            threshold = 0.35
            prediction, probabilities = pipeline.predict_one(
                unknown_lyrics=lyrics,
                threshold=threshold,
                model=classifier,
            )
            # prediction = classifier(lyrics, hypothesis)
            score = prediction[0][1]["score"]  # Get the entailment score
            results.append(score)

        # Normalize scores to sum to 1
        results = np.array(results)
        results = results / results.sum()

    return dict(zip(GENRES, results))


# Header
st.markdown(
    '<h1 class="main-header">Lyric Vision Spectrum Show</h1>', unsafe_allow_html=True
)
st.markdown(
    '<h2 class="sub-header">Discover the genre of your favorite lyrics through AI</h2>',
    unsafe_allow_html=True,
)

print("Loading classifier")
# Load the model
pipeline, model = load_model()

# Run this only once, not inside load_model()
webbrowser.open("http://localhost:8501/")


print("Classifier loaded successfully.")
# Main content
col1, col2 = st.columns([3, 2])

print("PLEASE WAIT UNTIL YOU SEE => INFO: Application startup complete.")

with col1:
    # Text area for lyrics input
    lyrics = st.text_area(
        "Paste your lyrics here",
        height=300,
        placeholder="Enter song lyrics to classify...",
    )

    # Submit button
    if st.button("Classify Genre"):
        if not lyrics.strip():
            st.error("Please enter some lyrics to classify.")
        elif classifier is None:
            st.error("Model failed to load. Please try again later.")
        else:
            # Get classification results
            results = classify_lyrics(lyrics, classifier)

            if results:
                # Store the results in session state
                st.session_state.results = results
                st.session_state.analyzed_lyrics = lyrics

# Display results
with col2:
    if "results" in st.session_state:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(
            '<h3 class="genre-header">Genre Analysis</h3>', unsafe_allow_html=True
        )

        # Convert results to DataFrame for visualization
        results_df = pd.DataFrame(
            {
                "Genre": list(st.session_state.results.keys()),
                "Probability": list(st.session_state.results.values()),
            }
        ).sort_values("Probability", ascending=False)

        # Create bar chart with Plotly
        fig = px.bar(
            results_df,
            x="Genre",
            y="Probability",
            color="Probability",
            color_continuous_scale="purples",
            text_auto=".1%",
            title=f"Top Genre: {results_df.iloc[0]['Genre']}",
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Probability",
            yaxis_tickformat=".0%",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display top 3 genres
        st.subheader("Top 3 Genres:")
        top3 = results_df.head(3)
        for i, row in top3.iterrows():
            st.markdown(f"**{row['Genre']}**: {row['Probability']:.1%}")

        st.markdown("</div>", unsafe_allow_html=True)

# Add some information in the sidebar
with st.sidebar:
    st.title("About")
    st.write(
        """
    This app uses AI to analyze song lyrics and predict their music genre. 
    Paste your favorite lyrics and see what genre they most closely match!
    """
    )

    st.subheader("How it works")
    st.write(
        """
    1. Paste lyrics in the text box
    2. Click "Classify Genre" 
    3. View the genre distribution visualization
    """
    )

    st.subheader("Supported Genres")
    for genre in GENRES:
        st.write(f"- {genre}")


if pipeline:
    try:
        on_shutdown()
    except Exception as e:
        print(f"Error during shutdown: {e}")
