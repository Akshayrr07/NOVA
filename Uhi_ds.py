import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic UHI data
dates = pd.date_range(start='2000-01-01', end='2024-01-01', freq='M')
np.random.seed(42)  # For reproducibility

# Create random urban and rural temperature data with some trend
urban_temp = 25 + 0.05 * np.arange(len(dates)) + np.random.normal(0, 2, len(dates))
rural_temp = 23 + 0.03 * np.arange(len(dates)) + np.random.normal(0, 2, len(dates))

# Create a DataFrame for UHI dataset
uhi_data = pd.DataFrame({
    'date': dates,
    'urban_temp': urban_temp,
    'rural_temp': rural_temp
})

# Calculate UHI effect (urban temperature - rural temperature)
uhi_data['UHI_effect'] = uhi_data['urban_temp'] - uhi_data['rural_temp']

# Save the UHI data to a CSV file
uhi_data.to_csv('uhi_dataset.csv', index=False)

# Load the data from CSV to verify
uhi_data = pd.read_csv('uhi_dataset.csv')

# Preprocess the data (assuming 'urban_temp' and 'rural_temp' columns)
uhi_data['date'] = pd.to_datetime(uhi_data['date'])
uhi_data.set_index('date', inplace=True)

# Plot UHI effect over time
plt.figure(figsize=(10, 6))
plt.plot(uhi_data.index, uhi_data['UHI_effect'], label='UHI Effect')
plt.title('Urban Heat Island Effect Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature Differential (°C)')
plt.legend()
plt.show()
