from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# Load pre-trained models
sea_ice_model = joblib.load('MODELS/sea_ice_model.pkl')  # RandomForest model
with open('MODELS/uhi_model.pkl', 'rb') as file:
    uhi_model = pickle.load(file)  # Assuming trained separately

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Use .get() to avoid KeyError
    sea_ice_input = request.form.get('sea_ice_input')
    urban_temp = request.form.get('urban_temp')
    rural_temp = request.form.get('rural_temp')

    # Debugging prints
    print(f"Received sea_ice_input: {sea_ice_input}")
    print(f"Received urban_temp: {urban_temp}")
    print(f"Received rural_temp: {rural_temp}")

    # Check if form data is missing
    if not sea_ice_input or not urban_temp or not rural_temp:
        return jsonify({'error': 'Missing input data'}), 400

    try:
        # Convert inputs to the right format
        sea_ice_input = np.array(sea_ice_input.split(), dtype=float).reshape(1, -1)
        urban_temp = float(urban_temp)
        rural_temp = float(rural_temp)

        # Make predictions
        sea_ice_prediction = sea_ice_model.predict(sea_ice_input)
        uhi_effect = uhi_model.predict([[urban_temp, rural_temp]])

        # Debugging prints for predictions
        print(f"Sea ice prediction: {sea_ice_prediction}")
        print(f"UHI prediction: {uhi_effect}")

        return jsonify({
            'sea_ice_prediction': sea_ice_prediction.tolist(),
            'uhi_prediction': uhi_effect.tolist()
        })

    except ValueError as e:
        print(f"Error processing input: {e}")
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
