import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score, classification_report

df = pd.read_csv("tested_molecules_properties_all.csv")
print(df.head())

# List of specified parameters to use as features
selected_features = [
    'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'BalabanJ', 'BertzCT', 'Ipc', 'LabuteASA',
    'PEOE_VSA4', 'SlogP_VSA3', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA8', 'VSA_EState5',
    'FractionCSP3', 'NumAromaticRings', 'RingCount', 'fr_C_O', 'fr_C_O_noCOO', 'fr_thiazole'
]

# 1) Divide the dataset into X, Y for PKM2
smiles = df['SMILES']
df = df.drop('SMILES', axis=1)
X = df[selected_features]  # Select only the specified features
Y = df[['PKM2_inhibition', 'ERK2_inhibition']].copy(deep=True)

# Combination of the 2 target variables into a unique variable
Y_combined = Y['PKM2_inhibition'] * 2 + Y['ERK2_inhibition']
Y = Y_combined.map({0: 0, 1: 1, 2: 2, 3: 3})  # Map (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3
print(Y.head())

# Get class distribution
class_distribution = Counter(Y)
majority_class_count = class_distribution[0]

# 2) Split the dataset into training and test sets
if min(class_distribution.values()) > 1:
    X_train, X_test, Y_train, Y_test, smiles_train, smiles_test = train_test_split(
        X, Y, smiles,
        test_size=0.30,
        stratify=Y,  # Apply stratification
        random_state=190
    )
else:
    # No stratification if any combination class has only one instance
    X_train, X_test, Y_train, Y_test, smiles_train, smiles_test = train_test_split(
        X, Y, smiles,
        test_size=0.30,
        random_state=190
    )

class_distribution_before = Counter(Y_train)
print("Class distribution_before_down:", class_distribution_before)

# 4) Oversampling
majority_class = 0

X_train_majority = X_train[Y_train == majority_class]
Y_train_majority = Y_train[Y_train == majority_class]
X_train_minority = X_train[Y_train != majority_class]
Y_train_minority = Y_train[Y_train != majority_class]
print("X_minority", Counter(Y_train_minority))

# OVERSAMPLING at 50%
target_count = int(0.5 * majority_class_count)  # 50% of the majority class

# Oversample the minority classes to match 50% of the majority class
X_train_minority_resampled = resample(X_train_minority,
                                      replace=True,     # Sample with replacement
                                      n_samples=target_count,   # Number of samples to match 50% of the majority class
                                      random_state=42)  # Reproducible results
Y_train_minority_resampled = resample(Y_train_minority,
                                      replace=True,
                                      n_samples=target_count,
                                      random_state=42)

# Concatenate the resampled minority class with the majority class data
X_train_resampled = pd.concat([X_train_majority, X_train_minority_resampled])
Y_train_resampled = pd.concat([Y_train_majority, Y_train_minority_resampled])
class_distribution_over = Counter(Y_train_resampled)
print("Class distribution_after_over:", class_distribution_over)

# 5) Scale the data
scaler = StandardScaler()
X_train_resampled_PKM2_scaled = scaler.fit_transform(X_train_resampled)
X_test_PKM2_scaled = scaler.transform(X_test)

# 6) SVM
parameters = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf']
}

# Define the scoring metrics
scoring = {
    'accuracy': 'accuracy', 
    'f1_micro': 'f1_micro', 
    'balanced_accuracy': 'balanced_accuracy'
}

# Define a custom scorer for multiclass ROC AUC
roc_auc_ovr = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)

# Add the custom scorer to the scoring dictionary
scoring['roc_auc_ovr'] = roc_auc_ovr

# Initialize the SVM classifier
classifier = SVC(probability=True)

# Initialize GridSearchCV for SVM
gs = GridSearchCV(classifier, parameters, cv=2, scoring=scoring, verbose=90, n_jobs=-1, refit='accuracy')

# Fit the model on the resampled training data
gs.fit(X_train_resampled_PKM2_scaled, Y_train_resampled)

# Results
print(f"Best score: {gs.best_score_} using {gs.best_params_}")
print("Best parameters:", gs.best_params_)
print("Best cross-validation score:", gs.best_score_)

# PREDICTION
best_model = gs.best_estimator_
Y_pred_test = best_model.predict(X_test)
Y_pred_train = best_model.predict(X_train_resampled)


# Additional metrics
# Calculate F1-score, Precision, Recall, and accuracy for each class
accuracy = metrics.accuracy_score(Y_test, Y_pred_test)
bacc = metrics.balanced_accuracy_score(Y_test, Y_pred_test)
precision = metrics.precision_score(Y_test, Y_pred_test, average='macro')
recall = metrics.recall_score(Y_test, Y_pred_test, average='macro')
f1 = metrics.f1_score(Y_test, Y_pred_test, average='macro')

print(f"\ninhibition accuracy: {accuracy:.2f}")
print(f"balanced accuracy: {bacc:.2f}")
print(f"inhibition precision: {precision:.2f}")
print(f"inhibition recall: {recall:.2f}")
print(f"inhibition F1-score: {f1:.2f}")

# Detailed classification report
print("\nClassification Report:")
report = classification_report(Y_test, Y_pred_test, labels=[0, 1, 2, 3], target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(report)
