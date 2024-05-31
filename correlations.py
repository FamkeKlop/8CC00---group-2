import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# README
# This code computes the correlation matrix for the descriptors that are
# present in the specified csv file. It can also visualize this correlation
# matrix, but this becomes messy for a high number of descriptors. The code
# also determines the highest correlations based on a specified threshold value. 
# To this end, the code first removes self-correlations, duplicate correlations
# (i.e. the bottom/upper triangle of the correlation matrix), and correlations
# with "NaN" as their value.

# Get data from csv file
df = pd.read_csv("tested_molecules_properties_all.csv")

# Exclude the first three columns
df_descriptors = df.iloc[:, 3:]

# Compute the correlation matrix
corr_matrix = df_descriptors.corr()

# # Visualize the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix of Descriptors', fontweight='bold', fontsize=16)
# plt.show()

# Reformat the correlation matrix and remove self-correlations
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Descriptor1', 'Descriptor2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Descriptor1'] != corr_pairs['Descriptor2']]

# Add column with the absolute correlations
corr_pairs['AbsCorrelation'] = corr_pairs['Correlation'].abs()

# Drop correlations with NaN values
corr_pairs = corr_pairs.dropna(subset=['AbsCorrelation'])

# Sort Descriptor1 and Descriptor2 within each row to allow for duplicate removal
corr_pairs[['Descriptor1', 'Descriptor2']] = np.sort(corr_pairs[['Descriptor1', 'Descriptor2']], axis=1)

# Remove duplicate pairs
corr_pairs = corr_pairs.drop_duplicates(subset=['Descriptor1', 'Descriptor2'])

# Get highly correlated variables
threshold = 0.9 #np.percentile(corr_pairs['AbsCorrelation'], 90)
print(threshold)

high_corrs = corr_pairs[corr_pairs['AbsCorrelation'] >= threshold]

# Display the highest correlations
print(high_corrs)

