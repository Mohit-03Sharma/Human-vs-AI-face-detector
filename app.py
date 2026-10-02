import os
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import tensorflow as tf


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Human vs. AI Face Detector",
    page_icon="🤖",
    layout="wide"
)

# Slightly reduce unnecessary page padding so the app fits
# better within a single laptop viewport.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 1200px;
        }

        h1 {
            margin-bottom: 0.2rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 0.6rem;
            border-radius: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "best_mobilenet_finetuned.keras"
LABELS_PATH = "labels.csv"
IMAGE_DIR = "test_images"

NUM_ROUNDS = 10


# ============================================================
# Load model and metadata
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_metadata():
    return pd.read_csv(LABELS_PATH)


model = load_model()
labels_df = load_metadata()


# ============================================================
# Helper functions
# ============================================================

def preprocess_image(image_path):
    """
    Load and prepare an image for MobileNet inference.

    Important:
    The model was trained using image values in the [0, 255]
    range, so the image is NOT divided by 255 here.
    """

    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))

    arr = np.array(
        img,
        dtype=np.float32
    )

    arr = np.expand_dims(
        arr,
        axis=0
    )

    return img, arr


def predict_image(image_path):
    """
    Return:
        sigmoid score
        predicted label
    """

    _, arr = preprocess_image(image_path)

    score = float(
        model.predict(
            arr,
            verbose=0
        )[0][0]
    )

    pred = 1 if score >= 0.5 else 0

    return score, pred


def label_name(label):
    return (
        "Real"
        if int(label) == 1
        else "AI-Generated"
    )


def initialize_game():
    """
    Start a new balanced game.

    A 10-round game contains:
    - 5 Real images
    - 5 AI-generated images
    """

    half = NUM_ROUNDS // 2

    real_samples = labels_df[
        labels_df["label"] == 1
    ].sample(
        n=half,
        replace=False
    )

    ai_samples = labels_df[
        labels_df["label"] == 0
    ].sample(
        n=half,
        replace=False
    )

    sample = pd.concat(
        [real_samples, ai_samples]
    ).sample(
        frac=1
    ).reset_index(drop=True)

    st.session_state.game_data = sample
    st.session_state.round = 0
    st.session_state.human_score = 0
    st.session_state.model_score = 0
    st.session_state.answered = False
    st.session_state.last_result = None


def restart_game():
    initialize_game()


# ============================================================
# Session state
# ============================================================

if "game_data" not in st.session_state:
    initialize_game()


# ============================================================
# Header
# ============================================================

st.title("Human vs. AI Face Detector")

st.caption(
    "Can you beat a fine-tuned MobileNetV3Small at identifying "
    "real vs. AI-generated faces?"
)


# ============================================================
# Current game
# ============================================================

current_round = st.session_state.round


if current_round < NUM_ROUNDS:

    row = st.session_state.game_data.iloc[
        current_round
    ]

    image_path = os.path.join(
        IMAGE_DIR,
        row["filename"]
    )

    true_label = int(
        row["label"]
    )

    img, _ = preprocess_image(
        image_path
    )


    # --------------------------------------------------------
    # Round progress
    # --------------------------------------------------------

    progress_value = (
        current_round + 1
    ) / NUM_ROUNDS

    st.progress(
        progress_value
    )

    st.markdown(
        f"### Round {current_round + 1} / {NUM_ROUNDS}"
    )


    # --------------------------------------------------------
    # Main two-column layout
    # --------------------------------------------------------

    left_col, right_col = st.columns(
        [1.15, 1],
        gap="large"
    )


    # ========================================================
    # LEFT COLUMN - IMAGE
    # ========================================================

    with left_col:

        st.image(
            img,
            caption="Real or AI-Generated?",
            width=410
        )


    # ========================================================
    # RIGHT COLUMN - GAME CONTROLS
    # ========================================================

    with right_col:

        st.markdown("### Scoreboard")

        score_col1, score_col2 = st.columns(2)

        with score_col1:

            st.metric(
                "Human",
                st.session_state.human_score
            )

        with score_col2:

            st.metric(
                "Model",
                st.session_state.model_score
            )


        st.markdown("### Your Guess")

        guess_col1, guess_col2 = st.columns(2)

        user_guess = None


        # ----------------------------------------------------
        # Real button
        # ----------------------------------------------------

        with guess_col1:

            if st.button(
                "Real",
                use_container_width=True,
                disabled=st.session_state.answered,
                type="primary"
            ):

                user_guess = 1


        # ----------------------------------------------------
        # AI-generated button
        # ----------------------------------------------------

        with guess_col2:

            if st.button(
                "AI-Generated",
                use_container_width=True,
                disabled=st.session_state.answered
            ):

                user_guess = 0


        # ====================================================
        # Process user answer
        # ====================================================

        if user_guess is not None:

            model_score, model_pred = predict_image(
                image_path
            )

            human_correct = (
                user_guess == true_label
            )

            model_correct = (
                model_pred == true_label
            )


            if human_correct:
                st.session_state.human_score += 1

            if model_correct:
                st.session_state.model_score += 1


            st.session_state.answered = True

            st.session_state.last_result = {
                "true_label": true_label,
                "user_guess": user_guess,
                "model_pred": model_pred,
                "model_score": model_score,
                "human_correct": human_correct,
                "model_correct": model_correct
            }

            st.rerun()


        # ====================================================
        # Display round result
        # ====================================================

        if st.session_state.answered:

            result = st.session_state.last_result

            true_name = label_name(
                result["true_label"]
            )

            user_name = label_name(
                result["user_guess"]
            )

            model_name = label_name(
                result["model_pred"]
            )


            st.markdown("### Result")


            # Human result
            if result["human_correct"]:

                st.success(
                    f"You: {user_name} ✓"
                )

            else:

                st.error(
                    f"You: {user_name} ✗"
                )


            # Model result
            if result["model_correct"]:

                st.success(
                    f"Model: {model_name} ✓"
                )

            else:

                st.error(
                    f"Model: {model_name} ✗"
                )


            st.caption(
                f"Correct answer: {true_name}"
            )

            st.caption(
                "Model Real-probability score: "
                f"{result['model_score']:.3f}"
            )


            # ------------------------------------------------
            # Next round
            # ------------------------------------------------

            if st.button(
                "Next Image",
                use_container_width=True
            ):

                st.session_state.round += 1
                st.session_state.answered = False
                st.session_state.last_result = None

                st.rerun()


# ============================================================
# Final result screen
# ============================================================

else:

    st.progress(1.0)

    st.markdown("## Final Result")

    human_score = (
        st.session_state.human_score
    )

    model_score = (
        st.session_state.model_score
    )


    final_col1, final_col2 = st.columns(2)

    with final_col1:

        st.metric(
            "Human",
            f"{human_score}/{NUM_ROUNDS}"
        )

    with final_col2:

        st.metric(
            "Model",
            f"{model_score}/{NUM_ROUNDS}"
        )


    if human_score > model_score:

        st.success(
            "You beat the model!"
        )

    elif model_score > human_score:

        st.info(
            "The model wins!"
        )

    else:

        st.warning(
            "It's a tie!"
        )


    if st.button(
        "Play Again",
        use_container_width=True,
        type="primary"
    ):

        restart_game()

        st.rerun()


# ============================================================
# About the model
# ============================================================

with st.expander(
    "About the model"
):

    st.write(
        """
        The computer opponent uses the **fine-tuned
        MobileNetV3Small** classifier developed in this project.

        The model performs binary classification:

        - **0 = AI-Generated**
        - **1 = Real**

        Its output layer uses a sigmoid activation.

        A score of **0.5 or greater** is classified as Real,
        while a score below **0.5** is classified as
        AI-Generated.
        """
    )