import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import warnings
warnings.filterwarnings("ignore")

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl not found. Make sure you've saved your model.")

# Defining the features for the form (for reference)
FEATURE_NAMES = [
    "Age", "Sex (1=Male, 0=Female)", "CP (Chest Pain Type)", 
    "Trestbps (Resting BP)", "Chol (Cholesterol)", "Fbs (>120 mg/dl)", 
    "Restecg", "Thalach (Max HR)", "Exang (Exercise Angina)", 
    "Oldpeak", "Slope", "Ca ( Major Vessels)", "Thal (Thalassemia)"
]

# --- Routing for the Home Page (Input Form) ---
@app.route('/')
def home():
    # Pass the feature names to the HTML template for labels
    return render_template('index.html', feature_names=FEATURE_NAMES)

# --- Routing for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    # Geting all form values as a list of strings
    form_values = [x for x in request.form.values()]
    
    # Converting all values to float for the model
    try:
        final_features = [float(x) for x in form_values]
    except ValueError:
        return render_template('result.html', prediction_text="Error: All inputs must be numerical values.")

    # Converting the list to a numpy array and reshape it for the model (1 row, 13 columns)
    final_features_array = np.array(final_features).reshape(1, -1)

    # Making the prediction
    prediction = model.predict(final_features_array)
    
    # Determining the output message
    if prediction[0] == 0:
        output = "Good News! The patient does not have heart disease"
    else:
        output = "The Patient should visit the doctor"

    # Rendering the result page with the prediction
    return render_template('result.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)