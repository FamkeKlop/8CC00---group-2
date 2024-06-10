# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:05:34 2024

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
import matplotlib.pyplot as plt

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
    
    return pca, n_pcs_needed, dataset_scaled, explained_variance_ratio

    """
    Identifies the features with the lowest absolute loadings for each principal component.

    Parameters:
    loadings (array-like): The loadings of the principal components.
    feature_names (list): The names of the features.
    n_bottom (int): The number of bottom features to select for each principal component. Default is 5.
    n_components (int): The number of principal components to consider. Default is 10.

    Returns:
    bottom_features (list of lists): A list where each sublist contains the bottom features for a principal component.
    """
    bottom_features = []
    
    for i in range(n_components):
        component_loadings = loadings[i]
        if len(component_loadings) <= n_bottom:
            bottom_indices = np.arange(len(component_loadings))
        else:
            bottom_indices = np.argpartition(np.abs(component_loadings), n_bottom)[:n_bottom]
        bottom_features.append([feature_names[idx] for idx in bottom_indices])
    
    return bottom_features

df = pd.read_csv("tested_molecules_properties_all.csv")

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

# ERK2 features
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

# 1) Divide the dataset into X, Y for PKM2
smiles = df['SMILES']
df = df.drop('SMILES', axis=1)
X = df.drop(['PKM2_inhibition', 'ERK2_inhibition'], axis=1)
Y_PKM2 = df['PKM2_inhibition']

# 2) Separate PKM2 and ERK2 inhibition data
X_PKM2 = X[features_PKM2]
X_ERK2 = X[features_ERK2]

# Scale and apply PCA on the PKM2 dataset
pca_train_PKM2, n_pcs_needed_PKM2, train_scaled_PKM2, loadings_PKM2 = scale_and_pca(X_PKM2)

# Scale and apply PCA on the ERK2 dataset
pca_train_ERK2, n_pcs_needed_ERK2, train_scaled_ERK2, loadings_ERK2 = scale_and_pca(X_ERK2)

