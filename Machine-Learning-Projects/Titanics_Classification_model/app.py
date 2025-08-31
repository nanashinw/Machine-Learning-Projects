from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
def load_titanic_model():
    """Load the trained Titanic survival prediction model"""
    try:
        # Try to load pickle model first
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            # If pickle doesn't exist, you can create a simple model for demonstration
            print("Model file not found. Please ensure model.pkl exists in the same directory.")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model at startup
model = load_titanic_model()

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None:
            return render_template('index.html', 
                                 result="Error: Model not loaded. Please check if model.pkl exists.")
        
        # Get form data
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        
        # Create feature array matching your model's expected input
        # Order: [Pclass, Sex, Age, SibSp, Parch]
        features = np.array([[pclass, sex, age, sibsp, parch]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            # For sklearn models
            probability = model.predict_proba(features)[0]
            predicted_class = model.predict(features)[0]
        else:
            # For neural network models (TensorFlow/Keras)
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Multi-class output
                predicted_class = np.argmax(prediction, axis=1)[0]
            else:
                # Binary classification
                predicted_class = 1 if prediction[0] > 0.5 else 0
        
        # Convert prediction to readable result
        result = "Survived" if predicted_class == 1 else "Did Not Survive"
        
        return render_template('index.html', result=result)
        
    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        return render_template('index.html', result=error_message)

@app.route('/about')
def about():
    """About page with model information"""
    return """
    <h1>About Titanic Survival Predictor</h1>
    <p>This web application uses machine learning to predict survival chances on the Titanic based on historical data.</p>
    <p>The model considers the following features:</p>
    <ul>
        <li><strong>Passenger Class (Pclass):</strong> 1st, 2nd, or 3rd class</li>
        <li><strong>Sex:</strong> Male (1) or Female (0)</li>
        <li><strong>Age:</strong> Age in years</li>
        <li><strong>Siblings/Spouses (SibSp):</strong> Number of siblings/spouses aboard</li>
        <li><strong>Parents/Children (Parch):</strong> Number of parents/children aboard</li>
    </ul>
    <p>The goal is to learn from historical maritime disasters to improve safety measures for future sea travel.</p>
    <p><a href="/">← Back to Predictor</a></p>
    """

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory. Please save the HTML template as 'templates/index.html'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)