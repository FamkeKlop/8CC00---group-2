# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:39:17 2024

@author: kbrus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
#from sklearn.svm import SVC
#from sklearn import metrics
from sklearn.decomposition import PCA

"""
Adjutsed code of Jort to include PCA, taking a explained variance of 75%. 
""" 
def scale_and_pca(dataset, variance_threshold=0.75):
    """
    Scales the data using StandardScaler, performs PCA on the data,
    and determines the number of principal components needed to capture a specified amount of variance.

    Parameters:
    dataset (array-like): The feature data.
    variance_threshold (float): The threshold of variance to capture. Default is 0.75 (75%).

    Returns:
    dataset_scaled (array-like): The scaled training data.
    pca (PCA object): The PCA object fitted on the scaled training data.
    n_pcs_needed (int): The number of principal components needed to capture the specified variance.
    """
    
    # Scale the resampled training data and the test data
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    
    # Perform PCA on the training set
    pca = PCA()
    pca.fit(dataset_scaled)
    
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Determine the number of principal components necessary to capture at least the specified variance
    summed_variance = np.cumsum(explained_variance_ratio)
    n_pcs_needed = np.argmax(summed_variance >= variance_threshold) + 1
    
    # Print the number of principal components needed
    print(f'Number of principal components needed to capture at least {variance_threshold*100}% of the variance: {n_pcs_needed}')
    
    return pca, n_pcs_needed, dataset_scaled



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

# 5) Scale and apply PCA on the training set PKM2
pca_train_PKM2, n_pcs_needed_PKM2, train_scaled_PKM2 = scale_and_pca(X_train_resampled_PKM2)