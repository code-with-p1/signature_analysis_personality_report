import os
import tensorflow as tf
# os.environ.pop("TF_USE_LEGACY_KERAS", None)
# os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from PIL import Image
from common.gradio.common import full_analysis

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Handwriting → Big Five Personality Prediction",
    layout="wide"
)

# ----------------------------
# Title & description
# ----------------------------
st.title("✍️ Handwriting → Big Five Personality Prediction")
st.markdown(
    "Upload any image of handwriting → model will try to predict personality traits."
)

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload handwriting image",
        type=["png", "jpg", "jpeg"]
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Handwriting Image", width="stretch")
    analyze_btn = st.button("Analyze", type="primary")

with col2:
    st.subheader("Prediction")
    prediction_placeholder = st.empty()

    st.subheader("Personality Description")
    summary_placeholder = st.empty()

# ----------------------------
# Inference logic
# ----------------------------
if analyze_btn and image is not None:
    with st.spinner("Analyzing handwriting..."):
        prediction, summary = full_analysis(image)

    prediction_placeholder.markdown(prediction)
    summary_placeholder.markdown(summary)

elif analyze_btn and image is None:
    st.warning("Please upload a handwriting image before clicking Analyze.")
