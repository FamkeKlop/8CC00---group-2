import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

# READ ME:
# This code prepares the dataset for the creation of ML model and applies KNN:
# 0) Apply correlation between featured to identify high-correlated features and remove one of them
# 1) separates the features (X) from the target variable (Y)
# 2) understands if the dataset is balanced
#   (since the target variable contains much more 0 (majority class) than 1 (minority classes)
#   the algorithm will not be able to represent the minority class
#   => I have to increase the number of observations whose Y is different from 0,
#   in order to make the dataset more balanced => oversampling)
# 3) splits the dataset into training and test set.
# 4) makes the 'oversampling' of the minority class in the training set (I shouldn't modify the test set)
# 5) Scaling and PCA
# 6) KNN
######################################
# 7) Results
# 8) Visualization


df = pd.read_csv("cleaned_df.csv")
print(df.head())
df_2=pd.read_csv("df_reduced.csv")

# 1) Divide the dataset into X, Y for PKM2

X_PKM2 = df.drop(['PKM2_inhibition','ERK2_inhibition','SMILES'], axis = 1)
X_PKM2 = df_2
X_ERK2 = X_PKM2
smiles = df['SMILES']
Y_PKM2 = df['PKM2_inhibition']
Y_ERK2 = df['ERK2_inhibition']

# 2) Understand if the dataset is balanced  # ---> 0.03 => imbalanced
class_distribution = Counter(Y_PKM2)
print("Class distribution:", class_distribution)
majority_class_count = class_distribution[0]
minority_classes_count = sum(count for cls, count in class_distribution.items() if cls != 0)
if minority_classes_count != 0:
    imbalance_ratio = minority_classes_count / majority_class_count
else:
    imbalance_ratio = -999  # error, the denominator can't be 0
print(f"Imbalance Ratio (majority:minority): {imbalance_ratio:.2f}")
print("\n")


# 3) Split the dataset into training and test sets for both PKM2 and ERK2
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
X_train_minority_resampled_ERK2 = resample(X_train_minority_ERK2, replace=True, n_samples=target_count_ERK2,
                                           random_state=42)
Y_train_minority_resampled_ERK2 = resample(Y_train_minority_ERK2, replace=True, n_samples=target_count_ERK2,
                                           random_state=42)

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
scaler_PKM2, pca_PKM2, n_pcs_needed_PKM2, X_train_resampled_PKM2_pca, X_test_PKM2_pca = scale_and_pca(
    X_train_resampled_PKM2, X_test_PKM2)

# Apply scaling and PCA on ERK2 dataset
scaler_ERK2, pca_ERK2, n_pcs_needed_ERK2, X_train_resampled_ERK2_pca, X_test_ERK2_pca = scale_and_pca(
    X_train_resampled_ERK2, X_test_ERK2)

# 6) KNN for PKM2
# Define the parameter grid
parameters_PKM2 = {'n_neighbors': np.arange(5, 20)}
scoring_PKM2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro', 'roc_auc': 'roc_auc','balanced_accuracy': 'balanced_accuracy'}
classifier_PKM2 = KNeighborsClassifier()
gs_PKM2 = GridSearchCV(classifier_PKM2, parameters_PKM2, cv=2, scoring=scoring_PKM2, verbose=90, n_jobs=-1, refit='balanced_accuracy')
gs_PKM2.fit(X_train_resampled_PKM2_pca, Y_train_resampled_PKM2)

# Results for PKM2
print(f"Best score for PKM2: {gs_PKM2.best_score_} using {gs_PKM2.best_params_}")
print("Best parameters for PKM2:", gs_PKM2.best_params_)
print("Best cross-validation score for PKM2:", gs_PKM2.best_score_)

# PREDICTION for PKM2
best_model_PKM2 = gs_PKM2.best_estimator_
Y_pred_test_PKM2 = best_model_PKM2.predict(X_test_PKM2_pca)
Y_pred_train_PKM2 = best_model_PKM2.predict(X_train_resampled_PKM2_pca)


# 6) KNN for ERK2
# Define the parameter grid
parameters_ERK2 = {'n_neighbors': np.arange(5, 20)}
scoring_ERK2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro', 'roc_auc': 'roc_auc','balanced_accuracy': 'balanced_accuracy'}
classifier_ERK2 = KNeighborsClassifier()
gs_ERK2 = GridSearchCV(classifier_ERK2, parameters_ERK2, cv=2, scoring=scoring_ERK2, verbose=90, n_jobs=-1, refit='balanced_accuracy')
gs_ERK2.fit(X_train_resampled_ERK2_pca, Y_train_resampled_ERK2)

# Results for ERK2
print(f"Best score for PKM2: {gs_ERK2.best_score_} using {gs_ERK2.best_params_}")
print("Best parameters for PKM2:", gs_ERK2.best_params_)
print("Best cross-validation score for PKM2:", gs_ERK2.best_score_)

# PREDICTION for ERK2
best_model_ERK2 = gs_ERK2.best_estimator_
Y_pred_test_ERK2 = best_model_ERK2.predict(X_test_ERK2_pca)
Y_pred_train_ERK2 = best_model_ERK2.predict(X_train_resampled_ERK2_pca)

# Saving scaler and PCA for PKM2
with open('scaler_PKM2.pkl', 'wb') as file:
    pickle.dump(scaler_PKM2, file)
with open('pca_PKM2.pkl', 'wb') as file:
    pickle.dump(pca_PKM2, file)

# Saving scaler and PCA for ERK2
with open('scaler_ERK2.pkl', 'wb') as file:
    pickle.dump(scaler_ERK2, file)
with open('pca_ERK2.pkl', 'wb') as file:
    pickle.dump(pca_ERK2, file)

# Saving models for PKM2 and ERK2
with open('model_KNN_PCA_PKM2.pkl', 'wb') as file:
    pickle.dump(best_model_PKM2, file)
with open('model_KNN_PCA_ERK2.pkl', 'wb') as file:
    pickle.dump(best_model_ERK2, file)


#####################################
# 8) Visualization

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

# 8) Visualization
# Confusion matrix for PKM2
cm_PKM2 = confusion_matrix(Y_test_PKM2, Y_pred_test_PKM2)
sns.heatmap(cm_PKM2, annot=True, fmt='d', cmap="Blues")
plt.title("PKM2 Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Confusion matrix for ERK2
cm_ERK2 = confusion_matrix(Y_test_ERK2, Y_pred_test_ERK2)
sns.heatmap(cm_ERK2, annot=True, fmt='d', cmap="Blues")
plt.title("ERK2 Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
