import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# READ ME:
# This code uses the original dataset as obtained after extracting all descriptors with
# the RDKit for each of the molecules present in the dataset provided with the assignment.
# This dataset is then used to predict the inhibition of PKM2 and ERK2 separately.
# To this end, the data is split into test and training data, after which oversampling 
# is used on the training data to compensate for the imbalance in the dataset. 
# Then, the data is scaled using standard scaling, and PCA is applied to the scaled data.
# Subsequently, grid search is used to find the optimal parameters for SVM that result 
# in the best score metrics. Finally, SVM is used to predict the kinase inhibition for
# the test set.

df = pd.read_csv("tested_molecules_properties_all.csv")


# 1) Divide the dataset into X, Y for both PKM2 and ERK2
smiles = df['SMILES']
X_PKM2 = df.drop(['SMILES','PKM2_inhibition', 'ERK2_inhibition'], axis=1)
X_ERK2 = df.drop(['SMILES','PKM2_inhibition', 'ERK2_inhibition'], axis=1)
Y_PKM2 = df['PKM2_inhibition']
Y_ERK2 = df['ERK2_inhibition']

# 2) Split the dataset into training and test sets for both PKM2 and ERK2
X_train_PKM2, X_test_PKM2, Y_train_PKM2, Y_test_PKM2, smiles_train_PKM2, smiles_test_PKM2 = train_test_split(
    X_PKM2, Y_PKM2, smiles, test_size=0.30, stratify=Y_PKM2, random_state=190
)

X_train_ERK2, X_test_ERK2, Y_train_ERK2, Y_test_ERK2, smiles_train_ERK2, smiles_test_ERK2 = train_test_split(
    X_ERK2, Y_ERK2, smiles, test_size=0.30, stratify=Y_ERK2, random_state=190
)

# 3a) Oversampling for PKM2
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

# 3b) Oversampling for ERK2
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

# 4) Scaling and PCA
def scale_and_pca(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=0.75)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    n_pcs_needed = pca.n_components_
    
    return scaler, pca, n_pcs_needed, X_train_pca, X_test_pca

# Apply scaling and PCA on PKM2 dataset
scaler_PKM2, pca_PKM2, n_pcs_needed_PKM2, X_train_resampled_PKM2_pca, X_test_PKM2_pca = scale_and_pca(X_train_resampled_PKM2, X_test_PKM2)

# Apply scaling and PCA on ERK2 dataset
scaler_ERK2, pca_ERK2, n_pcs_needed_ERK2, X_train_resampled_ERK2_pca, X_test_ERK2_pca = scale_and_pca(X_train_resampled_ERK2, X_test_ERK2)

# 5a) SVM for PKM2

# Define the parameter grid
parameters_PKM2 = {
    'C': list(range(1,11)),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf']
}

# Define the scoring metrics 
scoring_PKM2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro', 'roc_auc': 'roc_auc','balanced_accuracy': 'balanced_accuracy'}

# Initialize the SVM classifier
classifier_PKM2 = SVC(probability=True)

# Initialize GridSearchCV for SVM
gs_PKM2 = GridSearchCV(classifier_PKM2, parameters_PKM2, cv=2, scoring=scoring_PKM2, verbose=90, n_jobs=-1, refit='balanced_accuracy')

# Fit the model on the resampled training data
gs_PKM2.fit(X_train_resampled_PKM2_pca, Y_train_resampled_PKM2)

# Results for PKM2
print(f"Best score for PKM2: {gs_PKM2.best_score_} using {gs_PKM2.best_params_}")
print("Best parameters for PKM2:", gs_PKM2.best_params_)
print("Best cross-validation score for PKM2:", gs_PKM2.best_score_)

# Prediction for PKM2
best_model_PKM2 = gs_PKM2.best_estimator_
Y_pred_test_PKM2 = best_model_PKM2.predict(X_test_PKM2_pca)
Y_pred_train_PKM2 = best_model_PKM2.predict(X_train_resampled_PKM2_pca)

# 5b) SVM for ERK2

# Define the parameter grid
parameters_ERK2 = {
    'C': list(range(1,11)),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf']
}

# Define the scoring metrics 
scoring_ERK2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro', 'roc_auc': 'roc_auc','balanced_accuracy': 'balanced_accuracy'}

# Initialize the SVM classifier
classifier_ERK2 = SVC(probability=True)

# Initialize GridSearchCV for SVM
gs_ERK2 = GridSearchCV(classifier_ERK2, parameters_ERK2, cv=2, scoring=scoring_ERK2, verbose=90, n_jobs=-1, refit='balanced_accuracy')

# Fit the model on the resampled training data
gs_ERK2.fit(X_train_resampled_ERK2_pca, Y_train_resampled_ERK2)

# Results for ERK2
print(f"Best score for ERK2: {gs_ERK2.best_score_} using {gs_ERK2.best_params_}")
print("Best parameters for ERK2:", gs_ERK2.best_params_)
print("Best cross-validation score for ERK2:", gs_ERK2.best_score_)

# Prediction for ERK2
best_model_ERK2 = gs_ERK2.best_estimator_
Y_pred_test_ERK2 = best_model_ERK2.predict(X_test_ERK2_pca)
Y_pred_train_ERK2 = best_model_ERK2.predict(X_train_resampled_ERK2_pca)

# Print classification reports
print("\nPKM2:")
print(metrics.classification_report(Y_test_PKM2, Y_pred_test_PKM2))
print("\nERK2:")
print(metrics.classification_report(Y_test_ERK2, Y_pred_test_ERK2))

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


# Additional metrics
# Calculate F1-score, Precision, Recall, AUC-ROC, and accuracy for PKM2 inhibition
pkm2_accuracy = metrics.accuracy_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_bacc = metrics.balanced_accuracy_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_precision = metrics.precision_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_recall = metrics.recall_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_f1 = metrics.f1_score(Y_test_PKM2, Y_pred_test_PKM2)
pkm2_auc = metrics.roc_auc_score(Y_test_PKM2, Y_pred_test_PKM2)

print(f"\nPKM2 inhibition accuracy: {pkm2_accuracy:.2f}")
print(f"PKM2 balanced accuracy: {pkm2_bacc:.2f}")
print(f"PKM2 inhibition precision: {pkm2_precision:.2f}")
print(f"PKM2 inhibition recall: {pkm2_recall:.2f}")
print(f"PKM2 inhibition F1-score: {pkm2_f1:.2f}")
print(f"PKM2 inhibition AUC-ROC: {pkm2_auc:.2f}")

# Calculate F1-score, Precision, Recall, AUC-ROC, and accuracy for ERK2 inhibition
erk2_accuracy = metrics.accuracy_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_bacc = metrics.balanced_accuracy_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_precision = metrics.precision_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_recall = metrics.recall_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_f1 = metrics.f1_score(Y_test_ERK2, Y_pred_test_ERK2)
erk2_auc = metrics.roc_auc_score(Y_test_ERK2, Y_pred_test_ERK2)

print(f"\nERK2 inhibition accuracy: {erk2_accuracy:.2f}")
print(f"ERK2 balanced accuracy: {erk2_bacc:.2f}")
print(f"ERK2 inhibition precision: {erk2_precision:.2f}")
print(f"ERK2 inhibition recall: {erk2_recall:.2f}")
print(f"ERK2 inhibition F1-score: {erk2_f1:.2f}")
print(f"ERK2 inhibition AUC-ROC: {erk2_auc:.2f}")

# Calculate how often both kinases are predicted correctly
both_kinase_correct = ((final_df['PKM2_actual_inhibition'] == final_df['PKM2_pred_inhibition']) & 
                       (final_df['ERK2_actual_inhibition'] == final_df['ERK2_pred_inhibition'])).sum()
total_instances = len(final_df)
both_kinase_accuracy = both_kinase_correct / total_instances

print(f"\nBoth kinases correct (accuracy): {both_kinase_accuracy:.2f}")