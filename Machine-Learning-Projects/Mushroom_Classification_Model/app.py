import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Drug label mapping (based on your LabelEncoder)
drug_labels = {
    0: "Drug_drugA",
    1: "Drug_drugB", 
    2: "Drug_drugC",
    3: "Drug_drugX",
    4: "Drug_drugY"
}

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    try:
        # Get form data
        form_data = request.form
        
        # Create feature dictionary matching your model's expected columns
        # Based on pd.get_dummies() output from your training
        features_dict = {
            'Age': int(form_data.get('Age', 0)),
            'Na_to_K': float(form_data.get('Na_to_K', 0.0)),
            'Sex_F': 1 if form_data.get('Sex_F') == 'true' else 0,
            'Sex_M': 1 if form_data.get('Sex_M') == 'true' else 0,
            'BP_HIGH': 1 if form_data.get('BP_HIGH') == 'true' else 0,
            'BP_LOW': 1 if form_data.get('BP_LOW') == 'true' else 0,
            'BP_NORMAL': 1 if form_data.get('BP_NORMAL') == 'true' else 0,
            'Cholesterol_HIGH': 1 if form_data.get('Cholesterol_HIGH') == 'true' else 0,
            'Cholesterol_NORMAL': 1 if form_data.get('Cholesterol_NORMAL') == 'true' else 0
        }
        
        # Create DataFrame with the same structure as training data
        input_df = pd.DataFrame([features_dict])
        
        # Ensure all expected columns are present (in case some are missing)
        expected_columns = [
            'Age', 'Na_to_K', 'Sex_F', 'Sex_M', 'BP_HIGH', 
            'BP_LOW', 'BP_NORMAL', 'Cholesterol_HIGH', 'Cholesterol_NORMAL'
        ]
        
        # Add any missing columns with 0 values
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training order
        input_df = input_df[expected_columns]
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_label = drug_labels.get(prediction[0], f"Unknown Drug (Code: {prediction[0]})")
        
        # Debug: Print features for troubleshooting
        print(f"Input features: {features_dict}")
        print(f"DataFrame shape: {input_df.shape}")
        print(f"DataFrame values: {input_df.values}")
        print(f"Prediction: {prediction[0]}")
        
        return render_template("index.html", 
                             prediction_text=f"The recommended drug is: {prediction_label}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", 
                             prediction_text=f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    flask_app.run(debug=True)