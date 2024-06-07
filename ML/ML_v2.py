import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# READ ME:
# This code prepares the dataset for the creation of ML model:
# 1) separates the features (X) from the target variable (Y)
# 2) understand if the dataset is balanced
#   (since the target variable contains much more (0,0) (majority class) than (1,0),(0,1),(1,1) (minority classes)
#   the algorithm will not be able to represent the minority class
#   => I have to increase the number of observations whose Y is different from (0,0),
#   in order to make the dataset more balanced => oversampling)
# 3) splits the dataset into training and test set.
# 4) makes the 'oversampling' of the minority class in the training set (I shouldn't modify the test set)

df = pd.read_csv("tested_molecules_properties_all.csv")
print(df.head())

# List of specified parameters to use as features
selected_features = [
    'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'BalabanJ', 'BertzCT', 'Ipc', 'LabuteASA',
    'PEOE_VSA4', 'SlogP_VSA3', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA8', 'VSA_EState5',
    'FractionCSP3', 'NumAromaticRings', 'RingCount', 'fr_C_O', 'fr_C_O_noCOO', 'fr_thiazole'
]

# 1) Divide the dataset into X, Y
# Keep the SMILES column for the final dataframe
smiles = df['SMILES']

# drop the column SMILES
df = df.drop('SMILES', axis=1)
X = df[selected_features]  # Select only the specified features
Y = df[['PKM2_inhibition', 'ERK2_inhibition']].copy(deep=True)

# Combination of the 2 target variables into a unique variable
Y_combined = Y['PKM2_inhibition'] * 2 + Y['ERK2_inhibition']
Y = Y_combined.map({0: 0, 1: 1, 2: 2, 3: 3})  # Map (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3
print(Y.head())

# 2) Understand if the dataset is balanced
class_distribution = Counter(Y)
print("Class distribution:", class_distribution)  # I have much more (0,0) than the others
majority_class_count = class_distribution[0]
minority_classes_count = sum(count for cls, count in class_distribution.items() if cls != 0)
if minority_classes_count != 0:
    imbalance_ratio = minority_classes_count / majority_class_count
else:
    imbalance_ratio = -999  # error, the denominator can't be 0
print(f"Imbalance Ratio (majority:minority): {imbalance_ratio:.2f}")  # it is very low => imbalanced dataset
print("\n")

# 3) Split the dataset into training and test sets
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

# KNN
parameters = {'n_neighbors': np.arange(1, 20)}
scoring = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro'}

classifier = KNeighborsClassifier()
gs = GridSearchCV(classifier, parameters, cv=2, scoring=scoring, verbose=90, n_jobs=-1, refit='accuracy')
gs.fit(X_train_resampled, Y_train_resampled)

# Results
print(f"Best score: {gs.best_score_} using {gs.best_params_}")
print("Best parameters:", gs.best_params_)
print("Best cross-validation score:", gs.best_score_)

# PREDICTION
best_model = gs.best_estimator_
Y_pred_test = best_model.predict(X_test)
Y_pred_train = best_model.predict(X_train_resampled)

# Map combined predictions back to original variables
Y_pred_test_df = pd.DataFrame(Y_pred_test, columns=['Combined'])
Y_pred_test_df['PKM2_pred_inhibition'] = Y_pred_test_df['Combined'].map({0: 0, 1: 0, 2: 1, 3: 1})
Y_pred_test_df['ERK2_pred_inhibition'] = Y_pred_test_df['Combined'].map({0: 0, 1: 1, 2: 0, 3: 1})

# Original test labels for comparison
Y_test_df = pd.DataFrame(Y_test, columns=['Combined'])
Y_test_df['PKM2_actual_inhibition'] = Y_test_df['Combined'].map({0: 0, 1: 0, 2: 1, 3: 1})
Y_test_df['ERK2_actual_inhibition'] = Y_test_df['Combined'].map({0: 0, 1: 1, 2: 0, 3: 1})

# Combine SMILES, actual and predicted values into a final dataframe
final_df = pd.DataFrame({
    'SMILES': smiles_test.reset_index(drop=True),  # Ensure index alignment
    'PKM2_actual_inhibition': Y_test_df['PKM2_actual_inhibition'].reset_index(drop=True),
    'PKM2_pred_inhibition': Y_pred_test_df['PKM2_pred_inhibition'].reset_index(drop=True),
    'ERK2_actual_inhibition': Y_test_df['ERK2_actual_inhibition'].reset_index(drop=True),
    'ERK2_pred_inhibition': Y_pred_test_df['ERK2_pred_inhibition'].reset_index(drop=True)
})

# Print the final dataframe
print("\nFinal DataFrame with predictions and actual values:")
print(final_df.head())

# Evaluate your prediction
# test
f1_test = metrics.f1_score(Y_test, Y_pred_test, average='micro')
accuracy_test = metrics.accuracy_score(Y_test, Y_pred_test)
precision_test = metrics.precision_score(Y_test, Y_pred_test, average='weighted')
recall_test = metrics.recall_score(Y_test, Y_pred_test, average='weighted')

print("F1_score: %f" % (np.mean(f1_test)))          
print("accuracy: %f" % (np.mean(accuracy_test)))    
print("precision: %f" % (np.mean(precision_test)))  
print("recall: %f" % (np.mean(recall_test)))        

# Quick validation of accuracy
# Calculate the accuracy for PKM2 inhibition
pkm2_correct_predictions = (final_df['PKM2_actual_inhibition'] == final_df['PKM2_pred_inhibition']).sum()
pkm2_total_predictions = final_df.shape[0]
pkm2_accuracy = pkm2_correct_predictions / pkm2_total_predictions

# Calculate the accuracy for ERK2 inhibition
erk2_correct_predictions = (final_df['ERK2_actual_inhibition'] == final_df['ERK2_pred_inhibition']).sum()
erk2_total_predictions = final_df.shape[0]
erk2_accuracy = erk2_correct_predictions / erk2_total_predictions

print(f"PKM2 inhibition accuracy: {pkm2_accuracy:.2f}")
print(f"ERK2 inhibition accuracy: {erk2_accuracy:.2f}")

# Calculate the overall accuracy
overall_correct_predictions = (
    (final_df['PKM2_actual_inhibition'] == final_df['PKM2_pred_inhibition']) &
    (final_df['ERK2_actual_inhibition'] == final_df['ERK2_pred_inhibition'])
).sum()
overall_accuracy = overall_correct_predictions / pkm2_total_predictions

print(f"Overall accuracy: {overall_accuracy:.2f}")
