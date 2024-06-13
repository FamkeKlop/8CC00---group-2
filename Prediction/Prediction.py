import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('untested_molecules_properties_all.csv')
print(df.shape)

smiles = df["SMILES"]
df_features = pd.read_csv('df_reduced.csv')
X_final = df[df_features.columns]

# 5) Scaling
scaler = StandardScaler()
X_final=scaler.fit_transform(X_final)

# open the best classification model
with open('model_KNN_PKM2.pkl', 'rb') as file:
    model_KNN_PKM2 = pickle.load(file)
with open('model_KNN_ERK2.pkl', 'rb') as file:
    model_KNN_ERK2 = pickle.load(file)

Y_final_PKM2 = model_KNN_PKM2.predict(X_final)
Y_final_ERK2 = model_KNN_ERK2.predict(X_final)
print("PKM2",Counter(Y_final_PKM2))
print("ERK2",Counter(Y_final_ERK2))

# Create final dataset
final_df = pd.DataFrame({
    'SMILES': smiles,
    'PKM2_inhibition': Y_final_PKM2,
    'ERK2_inhibition': Y_final_ERK2
})

final_df.to_csv('final_df.csv', index=False)





