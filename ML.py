import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# READ ME:
# This code prepare the dataset for the creation of ML model:
# 1) separates the features (X) from the target variable (Y)
# 2) understand if the dataset is balanced
#   (since the target variable contains much more (0,0) (majority class) than (1,0),(0,1),(1,1) (minority classes)
#   the algorithm will not be able to represent the minority class
#   => I have to increase the number of observations whose Y is different from (0,0),
#   in order to make the dataset more balanced => oversampling)
# 3) splits the dataset in training and test set.
# 4) makes the 'oversampling' of the minority class in the training set (I shouldn't modify the test set)


df = pd.read_csv("tested_molecules_properties_all.csv")
print(df.head())
# df_pred = pd.read_csv("untested_molecules-3.csv")   # I will import the untested_molecules_properties (when I generate it)
# print(df_pred.head())

# 1) Divide the datasets in X, Y
# drop the column smile
df = df.drop('SMILES', axis=1)
X = df.drop(['PKM2_inhibition', 'ERK2_inhibition'], axis=1)
Y = df[['PKM2_inhibition', 'ERK2_inhibition']].copy(deep=True)
#df_pred.drop(['SMILES'],axis=1)
#X_pred = df_pred.drop(['PKM2_inhibition','ERK2_inhibition'], axis=1)

# combination of the 2 target variables in a unique variable
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
print(f"Imbalance Ratio (majority:minority): {imbalance_ratio:.2f}")    # it is very low => imbalanced dataset
print("\n")


# 3) Split the dataset in training and test sets
if min(class_distribution.values()) > 1:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.30,
        stratify=Y['combined'],  # Apply stratification
        random_state=190
    )
else:
    # No stratification if any combination class has only one instance
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
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





# this is an idea of how to make a classificator

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

# Evaluate your prediction
# test
f1_test = metrics.f1_score(Y_test, Y_pred_test, average='micro')
accuracy_test = metrics.accuracy_score(Y_test, Y_pred_test)
precision_test = metrics.precision_score(Y_test, Y_pred_test, average='weighted')
recall_test = metrics.recall_score(Y_test, Y_pred_test, average='weighted')

print("F1_score: %f" % (np.mean(f1_test)))          # 0.86
print("accuracy: %f" % (np.mean(accuracy_test)))    # 0.86
print("precision: %f" % (np.mean(precision_test)))  # 0.87
print("recall: %f" % (np.mean(recall_test)))        # 0.86
