"""
Streamlit web application for AI-powered pneumonia detection.

Run with:
    streamlit run app.py

The app lets a user upload a chest X-ray image (JPEG / PNG) and displays:
  - The original image
  - Predicted label (NORMAL / PNEUMONIA)
  - Confidence score with a progress bar
  - A brief clinical disclaimer
"""

import os
import tempfile

import streamlit as st
import tensorflow as tf

from pneumonia_detection import config
from pneumonia_detection.predict import predict_image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Model loading (cached so it is only loaded once)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model() -> tf.keras.Model | None:
    if not os.path.exists(config.MODEL_PATH):
        return None
    return tf.keras.models.load_model(config.MODEL_PATH)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🫁 AI-Powered Pneumonia Detection")
st.subheader("Chest X-Ray Classification using Deep Learning (ResNet50)")

st.markdown(
    """
Upload a chest X-ray image and the model will predict whether it shows
signs of **Pneumonia** or is **Normal**.

> ⚠️ **Disclaimer:** This tool is for research and educational purposes
> only. It is **not** a substitute for professional medical diagnosis.
> Always consult a qualified healthcare provider for medical advice.
"""
)

# ---------------------------------------------------------------------------
# Sidebar — model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(
        """
**Architecture:** ResNet50 (transfer learning)

**Training data:** Chest X-Ray Images (Pneumonia) — Kaggle

**Classes:**
- ✅ NORMAL
- 🔴 PNEUMONIA

**Input size:** 224 × 224 px (RGB)

**Reference:**  
[Kaggle dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
"""
    )

# ---------------------------------------------------------------------------
# Upload widget
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)

    # Save to a temp file so Keras can load it
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    model = load_model()

    if model is None:
        st.error(
            "⚠️ Trained model not found. "
            f"Please train the model first and save it to `{config.MODEL_PATH}`."
        )
    else:
        with st.spinner("Analysing image …"):
            result = predict_image(tmp_path, model=model)

        label = result["label"]
        confidence = result["confidence"]
        probability = result["probability"]

        # Colour-code the result
        if label == "PNEUMONIA":
            st.error(f"### 🔴 Prediction: **{label}**")
        else:
            st.success(f"### ✅ Prediction: **{label}**")

        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
        st.progress(confidence)

        with st.expander("Show raw probability"):
            st.write(f"P(PNEUMONIA) = {probability:.4f}")
            st.write(f"P(NORMAL)    = {1 - probability:.4f}")

    # Clean up temp file
    os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Developed by **Muhammad Abid Hussain** — NUST, Quetta, Pakistan | "
    "Research: EEG Signal Processing, Deep Learning, AI-Driven Healthcare"
)
