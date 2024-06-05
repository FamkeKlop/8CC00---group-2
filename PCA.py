from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


infile = 'tested_molecules_properties_all.csv'
# Load data from CSV file into a DataFrame
data = pd.read_csv(infile)

# Print top few rows of data
print(data.head())

# Select only the numerical features for standardization
numerical_features = data.drop(columns=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])


# Step 1: Standardize the numerical features
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)


# Step 2: Apply PCA
#pca = PCA(n_components=2)
pca = PCA(n_components=0.90)  # Retain components explaining 95% of the variance
pca.fit_transform(numerical_features_scaled)

# Step 3: Check the PCA components
#print("PCA components:", pca.components_)

# Prints how much % of the variance is captured in the dataset
print('Percentage of variance in dataset: ',sum(pca.explained_variance_ratio_))


nums = np.arange(len(numerical_features.columns))

var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(numerical_features_scaled)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))


#Creates a plot that shows the number of PCA components, compared to how much % of the variance is captured

plt.figure(figsize=(20,10),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')
plt.show()


# Step 6: Analyze principal component loadings
loadings_df = pd.DataFrame(pca.components_, columns=numerical_features.columns)
print("Principal Component Loadings:")
print(loadings_df)

#Creates csv file from loadings_df to check the values
loadings_df.to_csv('output.csv', index=False)

# Select a threshold for loading values (e.g., 0.1)
threshold = 0.01

# Identify features with loadings above or below the threshold for all principal components
selected_features = []
for column in loadings_df.columns:
    is_selected = True
    for value in loadings_df[column]:
        if abs(value) < threshold:
            is_selected = False
            break
    if is_selected:
        selected_features.append(column)


# Remove features that don't meet the threshold
filtered_numerical_features = numerical_features[selected_features]


# Print the selected features
print("Selected Features: ",filtered_numerical_features.columns)
print(" ")
