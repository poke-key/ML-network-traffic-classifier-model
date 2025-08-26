import sys
import pandas as pd
import joblib
import json
import os

def predict_traffic(input_file_path):
    """
    Load the ML model and make predictions on the input CSV file
    """
    try:
        # Load the model and scaler
        # Look for models in the current directory first, then parent directory
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_tuned_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
        
        # If not found in current directory, try parent directory
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'svm_tuned_model.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return {
                'error': 'Model files not found. Please ensure svm_tuned_model.pkl and scaler.pkl are in the models directory.'
            }
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Read the CSV file
        df = pd.read_csv(input_file_path)
        
        # Scale the features
        X_scaled = scaler.transform(df)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Map predictions to labels
        label_map = {
            0: "Streaming",
            1: "Secure", 
            2: "DNS",
            3: "Web",
            4: "Other"
        }
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'index': i,
                'category': int(pred),
                'label': label_map.get(int(pred), f"Class {pred}")
            })
        
        # Calculate category counts
        counts = {}
        for result in results:
            label = result['label']
            counts[label] = counts.get(label, 0) + 1
        
        category_counts = [{'category': k, 'count': v} for k, v in counts.items()]
        
        return {
            'predictions': results,
            'categoryCounts': category_counts,
            'message': 'Predictions completed successfully'
        }
        
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}'
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Usage: python predict.py <input_csv_file>'}))
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = predict_traffic(input_file)
    print(json.dumps(result)) 