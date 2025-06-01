import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/svm_tuned_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Network Traffic Classifier", layout="centered")

st.title("🔍 Network Traffic Classifier")
st.markdown("Upload a CSV file with flow-level features to predict traffic categories.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read and transform input
        df = pd.read_csv(uploaded_file)
        st.write("✅ File successfully loaded.")

        # Show preview
        st.subheader("Input Preview")
        st.dataframe(df.head())

        # Predict
        X_scaled = scaler.transform(df)
        preds = model.predict(X_scaled)

        st.subheader("Predicted Traffic Categories")
        st.write(preds.tolist())

        # show labeled value counts
        label_map = {
            0: "Streaming",
            1: "Secure",
            2: "DNS",
            3: "Web",
            4: "Other"
        }
        label_preds = [label_map.get(p, f"Class {p}") for p in preds]

        # Show labeled prediction counts
        pred_series = pd.Series(label_preds)
        st.bar_chart(pred_series.value_counts())

    except Exception as e:
        st.error(f"Error: {str(e)}")
