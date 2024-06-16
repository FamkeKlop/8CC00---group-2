import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler

# READ ME
# This code predicts the target variables for the untested_molecules data.
# It uses different ML models for the 2 molecules:
# PKM2: KNN-model with PCA
# ERK2: KNN-model with correlation
# At the end the final dataset (composed only of SMILES ad target variables) is generated.

df = pd.read_csv('untested_molecules_properties_all.csv')   # It contains all the columns
print(df.shape)
df_cleaned = pd.read_csv('cleaned_df.csv')
df_cleaned_features = df[df_cleaned.columns]    # selects all the columns of df_cleaned
smiles = df["SMILES"]

#####################
# PKM2
X_PKM2 = df_cleaned_features.drop(['SMILES','PKM2_inhibition', 'ERK2_inhibition'], axis=1)
# import ML model
with open('scaler_PKM2.pkl', 'rb') as file:
    scaler_PKM2 = pickle.load(file)
with open('model_KNN_PCA_PKM2.pkl', 'rb') as file:
    model_KNN_PKM2 = pickle.load(file)
with open('pca_PKM2.pkl', 'rb') as file:
    pca_PKM2 = pickle.load(file)
X_PKM2 = scaler_PKM2.transform(X_PKM2)
X_pca_PKM2 = pca_PKM2.transform(X_PKM2)
Y_final_PKM2 = model_KNN_PKM2.predict(X_pca_PKM2)   # Prediction
print("PKM2", Counter(Y_final_PKM2))

######################
# ERK2
df_features = pd.read_csv('df_reduced.csv')
X_final = df[df_features.columns]   # select the columns of df_reduced (after correlation)

scaler = StandardScaler()
X_final = scaler.fit_transform(X_final)
# import ML model
with open('model_KNN_ERK2.pkl', 'rb') as file:
    model_KNN_ERK2 = pickle.load(file)
Y_final_ERK2 = model_KNN_ERK2.predict(X_final)  # prediction
print("ERK2",Counter(Y_final_ERK2))


######################
# Create the final dataset
final_df = pd.DataFrame({
    'SMILES': smiles,
    'PKM2_inhibition': Y_final_PKM2,
    'ERK2_inhibition': Y_final_ERK2
})

final_df.to_csv('final_df.csv', index=False)





