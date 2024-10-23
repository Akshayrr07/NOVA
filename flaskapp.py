from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load pre-trained models
sea_ice_model = load_model('sea_ice_model.h5')  # Your saved sea ice LSTM model
uhi_model = LinearRegression()  # Your trained UHI regression model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Sea Ice Prediction
    sea_ice_input = np.array(data['sea_ice_input']).reshape(-1, 1)  # Input data from user
    sea_ice_prediction = sea_ice_model.predict(sea_ice_input)

    # UHI Prediction
    urban_temp = data['urban_temp']
    rural_temp = data['rural_temp']
    uhi_effect = uhi_model.predict([[urban_temp, rural_temp]])

    return jsonify({
        'sea_ice_prediction': sea_ice_prediction.tolist(),
        'uhi_prediction': uhi_effect.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
