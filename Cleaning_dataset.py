import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import IsolationForest

df = pd.read_csv("tested_molecules_properties_all.csv")
columns_to_convert = df.columns.difference(['PKM2_inhibition', 'ERK2_inhibition', 'SMILES'])
df[columns_to_convert] = df[columns_to_convert].astype(float)
print(df.dtypes)

# Remove column with the same value in all the rows
identical_value_columns = df.columns[df.nunique() == 1]
print("columns to remove:", identical_value_columns)
df = df.drop(columns=identical_value_columns)
print("Shape without columns:", df.shape)

# Remove identical columns
duplicate_columns = df.T.duplicated(keep='first')
columns_to_remove = df.columns[duplicate_columns]
df = df.T.drop_duplicates().T
print("Shape without duplicated columns:", df.shape)


X = df.drop(['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'], axis=1)
Y = df[['PKM2_inhibition', 'ERK2_inhibition']].copy(deep=True)
Y_combined = Y['PKM2_inhibition'] * 2 + Y['ERK2_inhibition']
Y = Y_combined.map({0: 0, 1: 1, 2: 2, 3: 3})  # Map (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3


# OUTLIERS
# Isolation Forest: ensemble-based outlier detection algorithm, to detect outliers in your dataset.
# It constructs isolation trees (binary decision trees) by randomly selecting a feature and then randomly selecting
# a split value between the maximum and minimum values of that feature.
# This process is repeated recursively until the data points are isolated into individual leaf nodes.
# The anomalies are expected to be isolated into shorter paths compared to normal points.

# Since using different seeds I may obtain slightly different outliers, I use 15 different seeds (randomly generated)
# and I select only the outliers that appear in the 75% of them
num_seeds = 15  # number of seeds
random_seeds = np.random.randint(1, 1000, size=num_seeds)  # create random seeds

outlier_sets = []   # contains outliers for each seed

# Isolation forest for each seed to obtain the outliers
for seed in random_seeds:
    isolation_forest = IsolationForest(n_estimators=700, random_state=seed)
    isolation_forest.fit(X)
    outlier_predictions = isolation_forest.predict(X)
    outlier_indices = np.where(outlier_predictions == -1)[0]
    outlier_sets.append(set(outlier_indices))

# Count how many times the outliers appear in the groups
outlier_counts = Counter(outlier for outliers in outlier_sets for outlier in outliers)
min_count = int(0.75 * num_seeds)    # I consider only outliers that appear in the 75% of the seeds
common_outliers = {outlier for outlier, count in outlier_counts.items() if count >= min_count}
print("Common Outliers Indices (present in at least 75% of results):", common_outliers)

df_cleaned = df.drop(common_outliers)

# Verify the new shape of the cleaned dataset
print("Cleaned dataset shape:", df_cleaned.shape)
df_cleaned.to_csv("cleaned_df.csv", index=False)


