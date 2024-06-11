import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

# Read the dataset
df = pd.read_csv("tested_molecules_properties_all.csv")

# Define features for PKM2 and ERK2
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
    "PEOE_VSA5",
    "SlogP_VSA3",
    "Chi1v",
    "SMR_VSA10",
    "BCUT2D_MWLOW",
    "SlogP_VSA5",
    "NumHDonors",
    "NHOHCount",
    "HeavyAtomMolWt",
    "NumAromaticHeterocycles",
    "SlogP_VSA6",
    "VSA_EState5",
    "EState_VSA3",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_furan",
    "MolWt",
    "ExactMolWt",
    "FractionCSP3",
    "BCUT2D_MRLOW",
    "VSA_EState10",
    "Ipc",
    "PEOE_VSA1",
    "NOCount",
    "PEOE_VSA14",
    "MinEStateIndex",
    "SlogP_VSA12",
    "VSA_EState2",
    "SlogP_VSA1",
    "Chi0v",
    "SlogP_VSA8",
    "LabuteASA",
    "PEOE_VSA4"
]

features_ERK2 = [
    "FpDensityMorgan1",
    "SMR_VSA9",
    "SlogP_VSA6",
    "SlogP_VSA8",
    "NumAromaticRings",
    "RingCount",
    "MolLogP",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_nitro",
    "fr_nitro_arom",
    "PEOE_VSA4",
    "VSA_EState6",
    "SlogP_VSA5",
    "NumAromaticCarbocycles",
    "fr_benzene",
    "fr_thiazole",
    "fr_amide",
    "VSA_EState5",
    "SlogP_VSA10",
    "EState_VSA7",
    "SlogP_VSA3",
    "BertzCT",
    "BCUT2D_LOGPHI",
    "SMR_VSA7",
    "BalabanJ",
    "FpDensityMorgan2",
    "SMR_VSA5",
    "EState_VSA2",
    "Ipc",
    "MolMR",
    "HeavyAtomMolWt",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_alkyl_halide",
    "FractionCSP3",
    "MolWt",
    "ExactMolWt",
    "Chi1",
    "fr_Ar_NH",
    "fr_Nhpyrrole",
    "LabuteASA",
    "EState_VSA4"
]

# Separate SMILES column
smiles = df['SMILES']
df = df.drop('SMILES', axis=1)

# Define X and Y for both PKM2 and ERK2
X = df.drop(['PKM2_inhibition', 'ERK2_inhibition'], axis=1)
Y_PKM2 = df['PKM2_inhibition']
Y_ERK2 = df['ERK2_inhibition']

# PKM2 feature extraction
X_PKM2 = X[features_PKM2]
X_ERK2 = X[features_ERK2]

# Split the dataset into training and test sets for both PKM2 and ERK2
X_train_PKM2, X_test_PKM2, Y_train_PKM2, Y_test_PKM2, smiles_train_PKM2, smiles_test_PKM2 = train_test_split(
    X_PKM2, Y_PKM2, smiles, test_size=0.30, stratify=Y_PKM2, random_state=190
)

X_train_ERK2, X_test_ERK2, Y_train_ERK2, Y_test_ERK2, smiles_train_ERK2, smiles_test_ERK2 = train_test_split(
    X_ERK2, Y_ERK2, smiles, test_size=0.30, stratify=Y_ERK2, random_state=190
)

# Oversampling for PKM2
majority_class_PKM2 = 0
X_train_majority_PKM2 = X_train_PKM2[Y_train_PKM2 == majority_class_PKM2]
Y_train_majority_PKM2 = Y_train_PKM2[Y_train_PKM2 == majority_class_PKM2]
X_train_minority_PKM2 = X_train_PKM2[Y_train_PKM2 != majority_class_PKM2]
Y_train_minority_PKM2 = Y_train_PKM2[Y_train_PKM2 != majority_class_PKM2]

target_count_PKM2 = len(Y_train_majority_PKM2)
X_train_minority_resampled_PKM2 = resample(X_train_minority_PKM2, replace=True, n_samples=target_count_PKM2, random_state=42)
Y_train_minority_resampled_PKM2 = resample(Y_train_minority_PKM2, replace=True, n_samples=target_count_PKM2, random_state=42)

X_train_resampled_PKM2 = pd.concat([X_train_majority_PKM2, X_train_minority_resampled_PKM2])
Y_train_resampled_PKM2 = pd.concat([Y_train_majority_PKM2, Y_train_minority_resampled_PKM2])

# Oversampling for ERK2
majority_class_ERK2 = 0
X_train_majority_ERK2 = X_train_ERK2[Y_train_ERK2 == majority_class_ERK2]
Y_train_majority_ERK2 = Y_train_ERK2[Y_train_ERK2 == majority_class_ERK2]
X_train_minority_ERK2 = X_train_ERK2[Y_train_ERK2 != majority_class_ERK2]
Y_train_minority_ERK2 = Y_train_ERK2[Y_train_ERK2 != majority_class_ERK2]

target_count_ERK2 = len(Y_train_majority_ERK2)
X_train_minority_resampled_ERK2 = resample(X_train_minority_ERK2, replace=True, n_samples=target_count_ERK2, random_state=42)
Y_train_minority_resampled_ERK2 = resample(Y_train_minority_ERK2, replace=True, n_samples=target_count_ERK2, random_state=42)

X_train_resampled_ERK2 = pd.concat([X_train_majority_ERK2, X_train_minority_resampled_ERK2])
Y_train_resampled_ERK2 = pd.concat([Y_train_majority_ERK2, Y_train_minority_resampled_ERK2])

# Scaling and PCA
def scale_and_pca(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    n_pcs_needed = pca.n_components_
    
    return scaler, pca, n_pcs_needed, X_train_pca, X_test_pca

# Apply scaling and PCA on PKM2 dataset
scaler_PKM2, pca_PKM2, n_pcs_needed_PKM2, X_train_resampled_PKM2_pca, X_test_PKM2_pca = scale_and_pca(X_train_resampled_PKM2, X_test_PKM2)

# Apply scaling and PCA on ERK2 dataset
scaler_ERK2, pca_ERK2, n_pcs_needed_ERK2, X_train_resampled_ERK2_pca, X_test_ERK2_pca = scale_and_pca(X_train_resampled_ERK2, X_test_ERK2)

# KNN for PKM2
parameters_PKM2 = {'n_neighbors': [4]}
scoring_PKM2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro'}

classifier_PKM2 = KNeighborsClassifier()
gs_PKM2 = GridSearchCV(classifier_PKM2, parameters_PKM2, cv=2, scoring=scoring_PKM2, verbose=90, n_jobs=-1, refit='accuracy')
gs_PKM2.fit(X_train_resampled_PKM2_pca, Y_train_resampled_PKM2)

# Results for PKM2
print(f"Best score for PKM2: {gs_PKM2.best_score_} using {gs_PKM2.best_params_}")
print("Best parameters for PKM2:", gs_PKM2.best_params_)
print("Best cross-validation score for PKM2:", gs_PKM2.best_score_)

# Prediction for PKM2
best_model_PKM2 = gs_PKM2.best_estimator_
Y_pred_test_PKM2 = best_model_PKM2.predict(X_test_PKM2_pca)
Y_pred_train_PKM2 = best_model_PKM2.predict(X_train_resampled_PKM2_pca)

# KNN for ERK2
parameters_ERK2 = {'n_neighbors': [4]}
scoring_ERK2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro'}

classifier_ERK2 = KNeighborsClassifier()
gs_ERK2 = GridSearchCV(classifier_ERK2, parameters_ERK2, cv=2, scoring=scoring_ERK2, verbose=90, n_jobs=-1, refit='accuracy')
gs_ERK2.fit(X_train_resampled_ERK2_pca, Y_train_resampled_ERK2)

# Results for ERK2
print(f"Best score for ERK2: {gs_ERK2.best_score_} using {gs_ERK2.best_params_}")
print("Best parameters for ERK2:", gs_ERK2.best_params_)
print("Best cross-validation score for ERK2:", gs_ERK2.best_score_)

# Prediction for ERK2
best_model_ERK2 = gs_ERK2.best_estimator_
Y_pred_test_ERK2 = best_model_ERK2.predict(X_test_ERK2_pca)
Y_pred_train_ERK2 = best_model_ERK2.predict(X_train_resampled_ERK2_pca)





### PART 3: METRICS ###
from sklearn.metrics import classification_report, confusion_matrix

# PKM2 Performance Evaluation
print("PKM2 Test Classification Report")
print(classification_report(Y_test_PKM2, Y_pred_test_PKM2))

print("PKM2 Train Classification Report")
print(classification_report(Y_train_resampled_PKM2, Y_pred_train_PKM2))

# ERK2 Performance Evaluation
print("ERK2 Test Classification Report")
print(classification_report(Y_test_ERK2, Y_pred_test_ERK2))

print("ERK2 Train Classification Report")
print(classification_report(Y_train_resampled_ERK2, Y_pred_train_ERK2))

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

# Extract dataframe of rows where an actual or predicted value of one is present for quick validation
final_df_ones = final_df[
    (final_df['PKM2_actual_inhibition'] == 1) |
    (final_df['PKM2_pred_inhibition'] == 1) |
    (final_df['ERK2_actual_inhibition'] == 1) |
    (final_df['ERK2_pred_inhibition'] == 1)
]

# Print the final dataframe with ones in the rows
print("\nFinal DataFrame containing ones with predictions and actual values:")
print(final_df_ones.head())

# Additional metrics
# Calculate F1-score, Precision, Recall, AUC-ROC, and accuracy for PKM2 inhibition
pkm2_accuracy = metrics.accuracy_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_precision = metrics.precision_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_recall = metrics.recall_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_f1 = metrics.f1_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_auc = metrics.roc_auc_score(Y_test_PKM2, Y_pred_test_PKM2)

print(f"\nPKM2 inhibition accuracy: {pkm2_accuracy:.2f}")
print(f"PKM2 inhibition precision: {pkm2_precision:.2f}")
print(f"PKM2 inhibition recall: {pkm2_recall:.2f}")
print(f"PKM2 inhibition F1-score: {pkm2_f1:.2f}")
print(f"PKM2 inhibition AUC-ROC: {pkm2_auc:.2f}")

# Calculate F1-score, Precision, Recall, AUC-ROC, and accuracy for ERK2 inhibition
erk2_accuracy = metrics.accuracy_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_precision = metrics.precision_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_recall = metrics.recall_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_f1 = metrics.f1_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_auc = metrics.roc_auc_score(Y_test_ERK2, Y_pred_test_ERK2)

print(f"\nERK2 inhibition accuracy: {erk2_accuracy:.2f}")
print(f"ERK2 inhibition precision: {erk2_precision:.2f}")
print(f"ERK2 inhibition recall: {erk2_recall:.2f}")
print(f"ERK2 inhibition F1-score: {erk2_f1:.2f}")
print(f"ERK2 inhibition AUC-ROC: {erk2_auc:.2f}")
