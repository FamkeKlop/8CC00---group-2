# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:11:22 2024

@author: kbrus
Source: https://biapol.github.io/blog/ryan_savill/principal_feature_analysis/readme.html#:~:text=Principal%20feature%20analysis%20(PFA)%20is,on%20this%20paper%20by%20Y.
"""
from sklearn import decomposition
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

class PFA(object):
    def __init__(self, diff_n_features = 2, q=None, explained_var = 0.95):
        self.q = q
        self.diff_n_features = diff_n_features
        self.explained_var = explained_var

    def fit(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA().fit(X)
        
        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
            for i,j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break
                    
        A_q = pca.components_.T[:,:q]
        
        clusternumber = min([q + self.diff_n_features, X.shape[1]])
        
        kmeans = KMeans(n_clusters= clusternumber).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        
    def fit_transform(self,X):    
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA().fit(X)
        
        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
            for i,j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break
                    
        A_q = pca.components_.T[:,:q]
        
        clusternumber = min([q + self.diff_n_features, X.shape[1]])
        
        kmeans = KMeans(n_clusters= clusternumber).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        
        return X[:, self.indices_]
    
    def transform(self, X):
        return X[:, self.indices_]

# Testing
infile = 'tested_molecules_properties_all.csv'
data = pd.read_csv(infile)

# Print top few rows of data
print(data.head())

# Select only the numerical features for standardization
numerical_features = data.drop(columns=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])

# Drop binary columns
binary_columns = []
for col in numerical_features.columns:
    if numerical_features[col].nunique() == 2 and all(x in [0, 1] for x in numerical_features[col].unique()):
        binary_columns.append(col)

# Remove binary columns
df_filtered = numerical_features.drop(columns=binary_columns)


pfa = PFA(diff_n_features=2, explained_var= 0.95)
pfa.fit_transform(df_filtered)

featurekeys = [df_filtered.keys().tolist()[i] for i in pfa.indices_]
featurekeys