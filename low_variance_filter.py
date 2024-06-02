# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:06:04 2024

@author: kbrus
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

"""
How might we correct for errors or duplicates...??
"""
threshold = 0.01  # Example threshold, adjust based on your data
start_data = 3 # In our case, this might be either 3 or 5

df = pd.read_csv('tested_molecules_properties.csv')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.iloc[:, start_data:])  # Exclude the first three columns for scaling

# Replace the scaled data in the original DataFrame
df.iloc[:, start_data:] = scaled_data

variances = df.iloc[:, start_data:].var()  # Exclude the first three columns for variance calculation

# Create a boolean mask for columns to keep
columns_to_keep = variances > threshold
columns_to_keep = pd.concat([pd.Series([True, True, True]), columns_to_keep])
filtered_df = df.loc[:, columns_to_keep.values]

# Save to a new CSV file if needed
filtered_df.to_csv('filtered_and_scaled_dataset.csv', index=False)

plt.figure(figsize=(10, 6))
variances.plot(kind='box')
plt.title('Boxplot of Variances for All Descriptors')
plt.ylabel('Variance')
plt.grid(True)
plt.show()

