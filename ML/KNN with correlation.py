import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve


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
# 5) Scaling
# 6) KNN
######################################
# 7) Results


df = pd.read_csv("cleaned_df.csv")
print(df["PKM2_inhibition"])
print(df["ERK2_inhibition"])


# 0) Correlation btw features
df_corr = df.drop(['SMILES',"PKM2_inhibition","ERK2_inhibition"], axis=1)
correlation_matrix = df_corr.corr()

# defined threshold for strong correlation
threshold = 0.8

# identify columns with a correlation higher than 0.8 and remove one of them
to_remove = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            to_remove.add(colname_j)

print(f"Colomns to remove: {to_remove}")
df_reduced = df_corr.drop(columns=to_remove)
df_reduced.to_csv("df_reduced.csv", index=False)


# 1) Divide the dataset into X, Y for PKM2
X_PKM2 = pd.read_csv("df_reduced.csv")
X_ERK2 = pd.read_csv("df_reduced.csv")
smiles = df['SMILES']
Y_PKM2 = df['PKM2_inhibition']

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

# 5) Scale the resampled training data and the test data
scaler = StandardScaler()
X_train_resampled_PKM2_scaled = scaler.fit_transform(X_train_resampled_PKM2)
X_test_PKM2_scaled = scaler.transform(X_test_PKM2)

# 6) KNN for PKM2
# Define the parameter grid
print(np.unique(Y_train_resampled_PKM2))
parameters_PKM2 = {'n_neighbors': np.arange(5, 20)}

# Define the scoring metrics
scoring_PKM2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro', 'roc_auc': 'roc_auc','balanced_accuracy': 'balanced_accuracy'}

# Initialize the classifier
classifier_PKM2 = KNeighborsClassifier()

# Initialize GridSearchCV
gs_PKM2 = GridSearchCV(classifier_PKM2, parameters_PKM2, cv=2, scoring=scoring_PKM2, verbose=90, n_jobs=-1, refit='balanced_accuracy')

# Fit the model on the resampled training data
gs_PKM2.fit(X_train_resampled_PKM2_scaled, Y_train_resampled_PKM2)

# Results for PKM2
print(f"Best score for PKM2: {gs_PKM2.best_score_} using {gs_PKM2.best_params_}")
print("Best parameters for PKM2:", gs_PKM2.best_params_)
print("Best cross-validation score for PKM2:", gs_PKM2.best_score_)

# PREDICTION for PKM2
best_model_PKM2 = gs_PKM2.best_estimator_
Y_pred_test_PKM2 = best_model_PKM2.predict(X_test_PKM2_scaled)
Y_pred_train_PKM2 = best_model_PKM2.predict(X_train_resampled_PKM2_scaled)


################################################
# Repeat the process for ERK2
# 1) Divide the dataset into X, Y for ERK2
Y_ERK2 = df['ERK2_inhibition']

# 2) Understand if the dataset is balanced # ---> 0.05 => imbalanced
class_distribution = Counter(Y_ERK2)
print("\n")
print("Class distribution:", class_distribution)
majority_class_count = class_distribution[0]
minority_classes_count = sum(count for cls, count in class_distribution.items() if cls != 0)
if minority_classes_count != 0:
    imbalance_ratio = minority_classes_count / majority_class_count
else:
    imbalance_ratio = -999  # error, the denominator can't be 0
print(f"Imbalance Ratio (majority:minority): {imbalance_ratio:.2f}")
print("\n")


# 3) Split the dataset into training and test sets for ERK2
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

# 5) Scale the resampled training data and the test data
scaler = StandardScaler()
X_train_resampled_ERK2_scaled = scaler.fit_transform(X_train_resampled_ERK2)
X_test_ERK2_scaled = scaler.transform(X_test_ERK2)

# 6) KNN for ERK2
parameters_ERK2 = {'n_neighbors': np.arange(5, 20)}
scoring_ERK2 = {'accuracy': 'accuracy', 'f1_micro': 'f1_micro', 'roc_auc': 'roc_auc','balanced_accuracy': 'balanced_accuracy', 'precision_micro': 'precision_micro'}
classifier_ERK2 = KNeighborsClassifier()

# Initialize GridSearchCV
gs_ERK2 = GridSearchCV(classifier_ERK2, parameters_ERK2, cv=2, scoring=scoring_ERK2, verbose=90, n_jobs=-1, refit='precision_micro')

# Fit the model on the resampled training data
gs_ERK2.fit(X_train_resampled_ERK2_scaled, Y_train_resampled_ERK2)

# Results for ERK2
print(f"Best score for ERK2: {gs_ERK2.best_score_} using {gs_ERK2.best_params_}")
print("Best parameters for ERK2:", gs_ERK2.best_params_)
print("Best cross-validation score for ERK2:", gs_ERK2.best_score_)

# PREDICTION for ERK2
best_model_ERK2 = gs_ERK2.best_estimator_
Y_pred_test_ERK2 = best_model_ERK2.predict(X_test_ERK2_scaled)
Y_pred_train_ERK2 = best_model_ERK2.predict(X_train_resampled_ERK2_scaled)

# Save the model for ERK2
with open('model_KNN_ERK2.pkl', 'wb') as file:
    pickle.dump(best_model_ERK2, file)

# Save the model for PKM2
with open('model_KNN_PKM2.pkl', 'wb') as file:
    pickle.dump(best_model_PKM2, file)



#########################################################################################
# 7) Final results
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

# Display Dual Kinase Accuracy
print(f"\nBoth kinases correct (accuracy): {both_kinase_accuracy:.2f}")

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



# Calculate AUC-ROC for PKM2
y_probs_PKM2 = best_model_PKM2.predict_proba(X_test_PKM2_scaled)[:,1]  # Predict probabilities for the positive class
fpr_PKM2, tpr_PKM2, thresholds_PKM2 = metrics.roc_curve(Y_test_PKM2, y_probs_PKM2)

plt.plot(fpr_PKM2, tpr_PKM2, label='PKM2 ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - PKM2')
plt.legend()
plt.show()



auc_PKM2 = metrics.roc_auc_score(Y_test_PKM2, y_probs_PKM2)
print('AUC for PKM2: %.2f' % auc_PKM2)

# Calculate AUC-ROC for ERK2
y_probs_ERK2 = best_model_ERK2.predict_proba(X_test_ERK2_scaled)[:,1]  # Predict probabilities for the positive class
fpr_ERK2, tpr_ERK2, thresholds_ERK2 = metrics.roc_curve(Y_test_ERK2, y_probs_ERK2)

plt.plot(fpr_ERK2, tpr_ERK2, label='ERK2 ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - ERK2')
plt.legend()
plt.show()

auc_ERK2 = metrics.roc_auc_score(Y_test_ERK2, y_probs_ERK2)
print('AUC for ERK2: %.2f' % auc_ERK2)

# other visualization
# - precision_recall_curve e average_precision_score di Scikit-Learn:
# Visualizza la precisione (positive predictive value) contro il recall (true positive rate)
# per diverse soglie di classificazione. È particolarmente utile per dataset sbilanciati.

# Precision-Recall Curve PKM2
Y_score_PKM2 = best_model_PKM2.predict_proba(X_test_PKM2_scaled)[:, 1]
precision, recall, _ = precision_recall_curve(Y_test_PKM2, Y_score_PKM2, pos_label=1)
average_precision = average_precision_score(Y_test_PKM2, Y_score_PKM2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()
# Precision-Recall Curve ERK2
Y_score_ERK2 = best_model_ERK2.predict_proba(X_test_ERK2_scaled)[:, 1]
precision, recall, _ = precision_recall_curve(Y_test_ERK2, Y_score_ERK2, pos_label=1)
average_precision = average_precision_score(Y_test_ERK2, Y_score_ERK2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

"""
# - learning_curve di Scikit-Learn
# Mostra l'andamento della performance del modello in funzione della dimensione del dataset di addestramento.
# È utile per capire se il modello beneficia di più dati o se soffre di overfitting/underfit

# learning curve PKM2
train_sizes, train_scores, test_scores = learning_curve(
    best_model_PKM2, X_train_PKM2, Y_train_PKM2, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calcola le medie e gli errori standard delle performance di training e test
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot della learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.title('Learning Curve PKM2')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()



# Genera la learning curve ERK2
train_sizes, train_scores, test_scores = learning_curve(
    best_model_ERK2, X_train_ERK2, Y_train_ERK2, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calcola le medie e gli errori standard delle performance di training e test
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot della learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.title('Learning Curve ERK2')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()
"""