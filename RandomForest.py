import pandas as pd
from sklearn.ensemble import RandomForestClassifier

infile = 'tested_molecules_properties_all.csv'

# Load data from CSV file into a DataFrame
data = pd.read_csv(infile)


# Removes smiles from data
data = data.drop(columns=['SMILES'])

# Define X and Y for PKM2 inhibition
X = data
YP = data['PKM2_inhibition']

# Train the Random Forest model for PKM2 inhibition
rfP = RandomForestClassifier(n_estimators=100, random_state=42)
rfP.fit(X, YP)

# Evaluate the model for PKM2 Inhibition
importances_PKM2 = rfP.feature_importances_
feature_names = X.columns
feature_importance_PKM2 = pd.DataFrame({'Feature': feature_names, 'Importance': importances_PKM2})

# Rank features by importance for PKM2 Inhibition
feature_importance_PKM2 = feature_importance_PKM2.sort_values(by='Importance', ascending=False)

# Save the DataFrame to a CSV file for PKM2 Inhibition
feature_importance_PKM2.to_csv('output_sorted_PKM2.csv', index=False)

# Define Y for ERK2 inhibition
YE = data['ERK2_inhibition']

# Train the Random Forest model for ERK2 inhibition
rfE = RandomForestClassifier(n_estimators=100, random_state=42)
rfE.fit(X, YE)

# Evaluate the model for ERK2 Inhibition
importances_ERK2 = rfE.feature_importances_
feature_importance_ERK2 = pd.DataFrame({'Feature': feature_names, 'Importance': importances_ERK2})

# Rank features by importance for ERK2 Inhibition
feature_importance_ERK2 = feature_importance_ERK2.sort_values(by='Importance', ascending=False)

# Save the DataFrame to a CSV file for ERK2 Inhibition
feature_importance_ERK2.to_csv('output_sorted_ERK2.csv', index=False)