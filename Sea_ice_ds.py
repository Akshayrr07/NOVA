import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for sea ice extent
dates = pd.date_range(start='1980-01-01', end='2024-01-01', freq='M')
np.random.seed(42)  # For reproducibility

# Create random sea ice extent data with a declining trend
extent = np.maximum(10 - 0.02 * np.arange(len(dates)) + np.random.normal(0, 0.5, len(dates)), 0)

# Create a DataFrame
sea_ice_data = pd.DataFrame({
    'date': dates,
    'extent': extent
})

# Save the DataFrame to a CSV file
sea_ice_data.to_csv('sea_ice_extent.csv', index=False)

# Plot sea ice extent over time
sea_ice_data['date'] = pd.to_datetime(sea_ice_data['date'])
sea_ice_data.set_index('date', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(sea_ice_data.index, sea_ice_data['extent'], label='Sea Ice Extent')
plt.title('Sea Ice Extent Over Time')
plt.xlabel('Year')
plt.ylabel('Extent (million km²)')
plt.legend()
plt.show()
