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
    sea_ice_input = request.form['sea_ice_input']
    urban_temp = float(request.form['urban_temp'])
    rural_temp = float(request.form['rural_temp'])

    # Sea Ice Prediction
    sea_ice_input = np.array(sea_ice_input.split(), dtype=float).reshape(1, -1)
    sea_ice_prediction = sea_ice_model.predict(sea_ice_input)

    # UHI Prediction
    uhi_effect = uhi_model.predict([[urban_temp, rural_temp]])

    return jsonify({
        'sea_ice_prediction': sea_ice_prediction.tolist(),
        'uhi_prediction': uhi_effect.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
