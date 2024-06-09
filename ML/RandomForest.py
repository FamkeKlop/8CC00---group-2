import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# READ ME:
# This code prepares the dataset for the creation of ML model:
# 1) separates the features (X) from the target variable (Y)
# 2) understands if the dataset is balanced
#   (since the target variable contains much more (0,0) (majority class) than (1,0),(0,1),(1,1) (minority classes)
#   the algorithm will not be able to represent the minority class
#   => I have to increase the number of observations whose Y is different from (0,0),
#   in order to make the dataset more balanced => oversampling)
# 3) splits the dataset into training and test set.
# 4) makes the 'oversampling' of the minority class in the training set (I shouldn't modify the test set)

df = pd.read_csv("tested_molecules_properties_all.csv")
print(df.head())

# PKM2 features
features_PKM2 = [
    "NumAromaticRings",
    "RingCount",
    "fr_thiazole",
    "BertzCT",
    "BalabanJ",
    "Chi4v",
    "AvgIpc",
    "Chi3v",
    "PEOE_VSA3",
    "Chi2v",
    "NumHAcceptors",
    "BCUT2D_MWHI",
    "VSA_EState1",
    "NumHeteroatoms",
    "VSA_EState9",
    "EState_VSA6",
    "BCUT2D_MRHI",
    "fr_sulfonamd",
    "fr_thiophene",
    "PEOE_VSA5"
]

# ERK2 features
features_ERK2 = [
    "NumAromaticRings",
    "RingCount",
    "SlogP_VSA6",
    "fr_C_O",
    "fr_C_O_noCOO",
    "SlogP_VSA8",
    "SMR_VSA9",
    "FpDensityMorgan1",
    "MolLogP",
    "fr_nitro",
    "fr_nitro_arom",
    "PEOE_VSA4",
    "VSA_EState6",
    "fr_thiazole",
    "SlogP_VSA5",
    "fr_benzene",
    "NumAromaticCarbocycles",
    "fr_amide",
    "VSA_EState5",
    "SlogP_VSA10"
]

# 1) Divide the dataset into X, Y for PKM2
smiles = df['SMILES']
df = df.drop('SMILES', axis=1)
X = df.drop(['PKM2_inhibition', 'ERK2_inhibition'], axis=1)
Y_PKM2 = df['PKM2_inhibition']

# 2) Separate PKM2 and ERK2 inhibition data
X_PKM2 = X[features_PKM2]
X_ERK2 = X[features_ERK2]

# 3) Split the dataset into training and test sets for PKM2
X_train_PKM2, X_test_PKM2, Y_train_PKM2, Y_test_PKM2, smiles_train_PKM2, smiles_test_PKM2 = train_test_split(
    X_PKM2, Y_PKM2, smiles, test_size=0.30, stratify=Y_PKM2, random_state=190
)

# 4) Oversampling for PKM2
majority_class_PKM2 = 0
X_train_majority_PKM2 = X_train_PKM2[Y_train_PKM2 == majority_class_PKM2]
Y_train_majority_PKM2 = Y_train_PKM2[Y_train_PKM2 == majority_class_PKM2]
X_train_minority_PKM2 = X_train_PKM2[Y_train_PKM2 != majority_class_PKM2]
Y_train_minority_PKM2 = Y_train_PKM2[Y_train_PKM2 != majority_class_PKM2]

target_count_PKM2 = len(Y_train_majority_PKM2)  # Match the number of majority class samples
X_train_minority_resampled_PKM2 = resample(X_train_minority_PKM2, replace=True, n_samples=target_count_PKM2, random_state=42)
Y_train_minority_resampled_PKM2 = resample(Y_train_minority_PKM2, replace=True, n_samples=target_count_PKM2, random_state=42)

X_train_resampled_PKM2 = pd.concat([X_train_majority_PKM2, X_train_minority_resampled_PKM2])
Y_train_resampled_PKM2 = pd.concat([Y_train_majority_PKM2, Y_train_minority_resampled_PKM2])

# Random Forest for PKM2
parameters_PKM2 = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

scoring_PKM2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro'}
classifier_PKM2 = RandomForestClassifier()

gs_PKM2 = GridSearchCV(classifier_PKM2, parameters_PKM2, cv=2, scoring=scoring_PKM2, verbose=90, n_jobs=-1, refit='accuracy')
gs_PKM2.fit(X_train_resampled_PKM2, Y_train_resampled_PKM2)

# Results for PKM2
print(f"Best score for PKM2: {gs_PKM2.best_score_} using {gs_PKM2.best_params_}")
print("Best parameters for PKM2:", gs_PKM2.best_params_)
print("Best cross-validation score for PKM2:", gs_PKM2.best_score_)

# PREDICTION for PKM2
best_model_PKM2 = gs_PKM2.best_estimator_
Y_pred_test_PKM2 = best_model_PKM2.predict(X_test_PKM2)
Y_pred_train_PKM2 = best_model_PKM2.predict(X_train_resampled_PKM2)

# Repeat the process for ERK2

# 1) Divide the dataset into X, Y for ERK2
Y_ERK2 = df['ERK2_inhibition']

# 2) Split the dataset into training and test sets for ERK2
X_train_ERK2, X_test_ERK2, Y_train_ERK2, Y_test_ERK2, smiles_train_ERK2, smiles_test_ERK2 = train_test_split(
    X_ERK2, Y_ERK2, smiles, test_size=0.30, stratify=Y_ERK2, random_state=190
)

# 4) Oversampling for ERK2
majority_class_ERK2 = 0
X_train_majority_ERK2 = X_train_ERK2[Y_train_ERK2 == majority_class_ERK2]
Y_train_majority_ERK2 = Y_train_ERK2[Y_train_ERK2 == majority_class_ERK2]
X_train_minority_ERK2 = X_train_ERK2[Y_train_ERK2 != majority_class_ERK2]
Y_train_minority_ERK2 = Y_train_ERK2[Y_train_ERK2 != majority_class_ERK2]

target_count_ERK2 = len(Y_train_majority_ERK2)  # Match the number of majority class samples
X_train_minority_resampled_ERK2 = resample(X_train_minority_ERK2, replace=True, n_samples=target_count_ERK2, random_state=42)
Y_train_minority_resampled_ERK2 = resample(Y_train_minority_ERK2, replace=True, n_samples=target_count_ERK2, random_state=42)

X_train_resampled_ERK2 = pd.concat([X_train_majority_ERK2, X_train_minority_resampled_ERK2])
Y_train_resampled_ERK2 = pd.concat([Y_train_majority_ERK2, Y_train_minority_resampled_ERK2])

# Random Forest for ERK2
parameters_ERK2 = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

scoring_ERK2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro'}
classifier_ERK2 = RandomForestClassifier()

gs_ERK2 = GridSearchCV(classifier_ERK2, parameters_ERK2, cv=2, scoring=scoring_ERK2, verbose=90, n_jobs=-1, refit='accuracy')
gs_ERK2.fit(X_train_resampled_ERK2, Y_train_resampled_ERK2)

# Results for ERK2
print(f"Best score for ERK2: {gs_ERK2.best_score_} using {gs_ERK2.best_params_}")
print("Best parameters for ERK2:", gs_ERK2.best_params_)
print("Best cross-validation score for ERK2:", gs_ERK2.best_score_)

# PREDICTION for ERK2
best_model_ERK2 = gs_ERK2.best_estimator_
Y_pred_test_ERK2 = best_model_ERK2.predict(X_test_ERK2)
Y_pred_train_ERK2 = best_model_ERK2.predict(X_train_resampled_ERK2)

# Print final results
print("\nPKM2:")
print(metrics.classification_report(Y_test_PKM2, Y_pred_test_PKM2))
print("\nERK2:")
print(metrics.classification_report(Y_test_ERK2, Y_pred_test_ERK2))

# Quick validation of accuracy
# Calculate the accuracy for PKM2 inhibition
pkm2_correct_predictions = (Y_test_PKM2 == Y_pred_test_PKM2).sum()
pkm2_total_predictions = len(Y_test_PKM2)
pkm2_accuracy = pkm2_correct_predictions / pkm2_total_predictions

# Calculate the accuracy for ERK2 inhibition
erk2_correct_predictions = (Y_test_ERK2 == Y_pred_test_ERK2).sum()
erk2_total_predictions = len(Y_test_ERK2)
erk2_accuracy = erk2_correct_predictions / erk2_total_predictions

print(f"\nPKM2 inhibition accuracy: {pkm2_accuracy:.2f}")
print(f"ERK2 inhibition accuracy: {erk2_accuracy:.2f}")

# Combine SMILES, actual and predicted values into a final dataframe for PKM2
final_df_PKM2 = pd.DataFrame({
    'SMILES': smiles_test_PKM2.reset_index(drop=True),  # Ensure index alignment
    'PKM2_actual_inhibition': Y_test_PKM2.reset_index(drop=True),
    'PKM2_pred_inhibition': Y_pred_test_PKM2
})

# Combine SMILES, actual and predicted values into a final dataframe for ERK2
final_df_ERK2 = pd.DataFrame({
    'SMILES': smiles_test_ERK2.reset_index(drop=True),  # Ensure index alignment
    'ERK2_actual_inhibition': Y_test_ERK2.reset_index(drop=True),
    'ERK2_pred_inhibition': Y_pred_test_ERK2
})

# Merge the two dataframes on 'SMILES' column
final_df = pd.merge(final_df_PKM2, final_df_ERK2, on='SMILES')

# Print the final dataframe
print("\nFinal DataFrame with predictions and actual values:")
print(final_df.head())