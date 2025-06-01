import argparse
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/svm_tuned_model.pkl")
scaler = joblib.load("models/scaler.pkl")

label_map = {0: "Streaming", 1: "Secure", 2: "DNS", 3: "Web", 4: "Other"}

def predict(file):
    df = pd.read_csv(file)
    X = scaler.transform(df)
    preds = model.predict(X)
    print("Predictions:")
    print(f"Predicted class: {preds[0]} → {label_map[preds[0]]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a CSV file")
    parser.add_argument("--file", type=str, required=True, help="Path to input CSV file with features")
    args = parser.parse_args()
    predict(args.file)
