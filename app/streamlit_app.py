import sys
import os
import streamlit as st
import torch

# ✅ Ensure project root is in Python path so "src" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.infer_wrapper import predict_sequence

# Page config
st.set_page_config(
    page_title="Protein Family Classifier",
    page_icon="🧬",
    layout="centered"
)

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"**Device in use:** `{DEVICE}`")

st.markdown(
    "<h1 style='text-align: center;'>🧬 Protein Family Classifier</h1>",
    unsafe_allow_html=True
)
st.write("Enter a protein sequence to predict its family.")

# Sequence input (forces uppercase)
sequence = st.text_area(
    "Protein sequence",
    placeholder="Enter amino acid sequence (e.g., GDTRPRFLWQLK...)"
).strip().upper()

if st.button("Predict Family"):
    if not sequence:
        st.error("⚠ Please enter a protein sequence before predicting.")
    else:
        try:
            # Ensure predict_sequence always returns (prediction, confidence)
            result = predict_sequence(sequence)

            if isinstance(result, tuple) and len(result) == 2:
                predicted_family, confidence = result
                st.success(f"**Predicted Family:** {predicted_family}")
                st.info(f"Confidence: {confidence*100:.2f}%")
            else:
                predicted_family = result if isinstance(result, str) else str(result)
                st.success(f"**Predicted Family:** {predicted_family}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

st.markdown("---")
st.caption("Powered by your protein classifier model")
