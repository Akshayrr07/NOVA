import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load sea ice extent dataset
data_url = 'sea_ice_extent.csv'  # Replace with actual data URL
sea_ice_data = pd.read_csv(data_url)

# Convert date column to datetime
sea_ice_data['date'] = pd.to_datetime(sea_ice_data['date'])
sea_ice_data.set_index('date', inplace=True)

# Normalize data
scaler = MinMaxScaler()
sea_ice_scaled = scaler.fit_transform(sea_ice_data['extent'].values.reshape(-1, 1))

# Prepare data for regression model
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X, y = create_dataset(sea_ice_scaled, time_step)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Rescale the predicted values back to the original scale
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the mean squared error
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sea_ice_data.index[-len(y_test):], y_test_rescaled, label='Actual Sea Ice Extent')
plt.plot(sea_ice_data.index[-len(y_test):], y_pred_rescaled, label='Predicted Sea Ice Extent')
plt.title('Sea Ice Extent Prediction using RandomForestRegressor')
plt.xlabel('Year')
plt.ylabel('Extent (million km²)')
plt.legend()
plt.show()

import joblib

# Save the trained RandomForestRegressor model
model_filename = 'random_forest_regressor_model.pkl'
joblib.dump(regressor, model_filename)
print(f'Model saved as {model_filename}')

# Load the saved model
loaded_model = joblib.load(model_filename)

# Use the loaded model to make predictions
y_pred_loaded = loaded_model.predict(X_test)

# Rescale and calculate metrics as before
y_pred_rescaled_loaded = scaler.inverse_transform(y_pred_loaded.reshape(-1, 1))
mse_loaded = mean_squared_error(y_test_rescaled, y_pred_rescaled_loaded)
print(f'Mean Squared Error (loaded model): {mse_loaded}')
