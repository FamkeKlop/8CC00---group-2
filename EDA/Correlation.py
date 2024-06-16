import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# READ ME:
# This code applies correlation between variables to remove high-correlated variables.
# We have tried 2 different approaches:
# 1) Correlation btw variables selected by T-test (different for the 2 Kinases)
# 2) Correlation btw all the variables (cleaned_df)

df = pd.read_csv("cleaned_df.csv")
print(df.head())

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
   # "fr_C_O_noCOO",
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
    "NumAromaticRings",
    "RingCount",
    "SlogP_VSA6",
     "fr_C_O",
    #"fr_C_O_noCOO",
    "SlogP_VSA8",
    "SMR_VSA9",
    "FpDensityMorgan1",
    "MolLogP",
    "fr_nitro",
    #"fr_nitro_arom",
    "PEOE_VSA4",
    "VSA_EState6",
    "fr_thiazole",
    "SlogP_VSA5",
    "fr_benzene",
    "NumAromaticCarbocycles",
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
    #"fr_Nhpyrrole",
    "LabuteASA",
    "EState_VSA4"

]


# CORRELATION BETWEEN VARIABLES
df_corr = df.drop(['SMILES'], axis=1)   # dataset for the correlation with all the variables
df_corr_PKM2 = df_corr[features_PKM2]          # Dataset for correlation with selected features PKM2
df_corr_ERK2 = df_corr[features_ERK2]          # Dataset for correlation with selected features ERK2
# apply correlation
correlation_matrix_PKM2 =df_corr_PKM2.corr()
correlation_matrix_ERK2 =df_corr_ERK2.corr()
correlation_matrix=df_corr.corr()

threshold = 0.8

# 1) Correlation btw variables selected by T-test (different for the 2 Kinases)
# Columns with correlation higher than Threshold for PKM2
to_remove_PKM2 = set()
for i in range(len(correlation_matrix_PKM2.columns)):
    for j in range(i + 1, len(correlation_matrix_PKM2.columns)):
        if abs(correlation_matrix_PKM2.iloc[i, j]) > threshold:
            colname_i = correlation_matrix_PKM2.columns[i]
            colname_j = correlation_matrix_PKM2.columns[j]
            to_remove_PKM2.add(colname_j)

# Columns with correlation higher than Threshold for ERK2
to_remove_ERK2 = set()
for i in range(len(correlation_matrix_ERK2.columns)):
    for j in range(i + 1, len(correlation_matrix_ERK2.columns)):
        if abs(correlation_matrix_ERK2.iloc[i, j]) > threshold:
            colname_i = correlation_matrix_ERK2.columns[i]
            colname_j = correlation_matrix_ERK2.columns[j]
            to_remove_ERK2.add(colname_j)

# 2) Correlation btw all the variables (cleaned_df)
# Columns with correlation higher than Threshold
to_remove = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            to_remove.add(colname_j)


########################################
print(f"Columns to remove PKM2: {to_remove_PKM2}")
print(f"Columns to remove ERK2: {to_remove_ERK2}")
print(f"Columns to remove: {to_remove}")

# Remove columns
df_reduced_PKM2 = df_corr_PKM2.drop(columns=to_remove_PKM2)
df_reduced_ERK2 = df_corr_ERK2.drop(columns=to_remove_ERK2)
df_reduced = df_corr.drop(columns=to_remove)

# Correlation matrix
# PKM2
correlation_matrix_reduced = df_reduced_PKM2.corr()
sns.clustermap(correlation_matrix_reduced, cmap='coolwarm', linewidths=.5, figsize=(10, 8))
plt.show()
# ERK2
correlation_matrix_reduced = df_reduced_ERK2.corr()
sns.clustermap(correlation_matrix_reduced, cmap='coolwarm', linewidths=.5, figsize=(10, 8))
plt.show()

# Save datasets
df_reduced_PKM2.to_csv("df_PKM2.csv", index=False)
df_reduced_ERK2.to_csv("df_ERK2.csv", index=False)
df_reduced.to_csv("df_reduced.csv", index=False)


