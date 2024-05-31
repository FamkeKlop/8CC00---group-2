import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# README
# This code creates a boxplot for all descriptors present in the specified csv file.
# To allow for better comparison of the boxplots, the data is first scaled using either
# min-max scaling or standardization (you can choose which scaler to use).
# Because there are over 200 descriptors, the code creates multiple figures that each
# contain 15 boxplots. Instead of plotting them, it creates a new folder called 
# 'boxplots' in the current working directory, where it saves all the created figures
# as .png images.

# Get data from csv file
df = pd.read_csv("tested_molecules_properties_all.csv")

# Exclude the first three columns
df_descriptors = df.iloc[:, 3:]

# scaler = StandardScaler() # Standardize
scaler = MinMaxScaler() # Normalize
df_standardized = pd.DataFrame(scaler.fit_transform(df_descriptors), columns=df_descriptors.columns)

# Number of descriptors
num_vars = df_standardized.shape[1]

# Determine grid size
cols = 5
rows = math.ceil(num_vars / 15)  # 15 plots per figure

# Create a folder to save figures
output_folder = "boxplots"
os.makedirs(output_folder, exist_ok=True)

# Iterate through each group of 15 descriptors
for fig_num in range(rows):
    start_idx = fig_num * 15  # Starting index for variables in this figure
    end_idx = min((fig_num + 1) * 15, num_vars)  # Ending index for variables in this figure
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    # Plot each descriptor
    for i, idx in enumerate(range(start_idx, end_idx)):
        col = df_standardized.columns[idx]
        sns.boxplot(y=df_standardized[col], ax=axes[i])
        axes[i].set_title(col, fontweight='bold', fontsize=10)

    # Hide any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # Save the figure in the output folder
    fig.savefig(os.path.join(output_folder, f"boxplots_{fig_num + 1}.png"))
    plt.close(fig)  # Close the figure to free memory


