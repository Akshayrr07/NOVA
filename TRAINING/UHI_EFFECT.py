import matplotlib.pyplot as plt
import pandas as pd
# Load UHI data (example dataset)
uhi_data_url = 'DATASETS/uhi_dataset.csv'  # Replace with actual dataset
uhi_data = pd.read_csv(uhi_data_url)

# Preprocess the data (assuming 'urban_temp' and 'rural_temp' columns)
uhi_data['date'] = pd.to_datetime(uhi_data['date'])
uhi_data.set_index('date', inplace=True)

# Calculate UHI effect (urban temperature - rural temperature)
uhi_data['UHI_effect'] = uhi_data['urban_temp'] - uhi_data['rural_temp']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Prepare features and labels
X = uhi_data[['urban_temp', 'rural_temp']]  # Features: urban and rural temperatures
y = uhi_data['UHI_effect']  # Label: UHI effect

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict UHI effect
y_pred = regressor.predict(X_test)

# Plot predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual UHI Effect')
plt.ylabel('Predicted UHI Effect')
plt.title('Actual vs Predicted UHI Effect')
plt.show()

import joblib

# Train the UHI model
uhi_model = LinearRegression()
uhi_model.fit(X_train, y_train)

# Save the model
joblib.dump(uhi_model, 'uhi_model.pkl')