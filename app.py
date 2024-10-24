from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle

app = Flask(__name__)

# Load pre-trained models
sea_ice_model = joblib.load('MODELS/sea_ice_model.pkl')  # Your saved RandomForest model
with open('MODELS/uhi_model.pkl', 'rb') as file:
    uhi_model = pickle.load(file)  # Assuming you've trained and fitted this model separately

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Use .get() to safely retrieve form data and avoid KeyError
    sea_ice_input = request.form.get('sea_ice_input')
    urban_temp = request.form.get('urban_temp')
    rural_temp = request.form.get('rural_temp')

    # Check if form data is missing and handle it
    if not sea_ice_input or not urban_temp or not rural_temp:
        return jsonify({'error': 'Missing input data'}), 400

    try:
        # Convert inputs to appropriate types
        sea_ice_input = np.array(sea_ice_input.split(), dtype=float).reshape(1, -1)
        urban_temp = float(urban_temp)
        rural_temp = float(rural_temp)

        # Sea Ice Prediction
        sea_ice_prediction = sea_ice_model.predict(sea_ice_input)

        # UHI Prediction
        uhi_effect = uhi_model.predict([[urban_temp, rural_temp]])

        return jsonify({
            'sea_ice_prediction': sea_ice_prediction.tolist(),
            'uhi_prediction': uhi_effect.tolist()
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)

print("sea_ice_prediction","uhi_effect")