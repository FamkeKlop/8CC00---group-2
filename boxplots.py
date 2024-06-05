import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# README
# This code creates 4 boxplots for all descriptors present in the specified csv file,
# i.e. the boxplots are grouped based on kinase inhibition.
# To allow for better comparison of the boxplots, the data is first scaled using either
# min-max scaling or standardization (you can choose which scaler to use).
# Because there are over 200 descriptors, the code creates multiple figures that each
# contain 15 subplots, each containing 4 boxplots. Instead of plotting them, it creates 
# a new folder called 'boxplots' in the current working directory, where it saves all 
# the created figures as .png images.

# Get data from csv file
df = pd.read_csv("tested_molecules_properties_all.csv")

# Exclude the first three columns
df_descriptors = df.iloc[:, 3:]

# Scale the data
scaler1 = MinMaxScaler() # Normalize
scaler2 = StandardScaler() # Standardize
df_scaled = pd.DataFrame(scaler1.fit_transform(df_descriptors), columns=df_descriptors.columns)

# Number of descriptors
num_vars = df_scaled.shape[1]

# Determine grid size
descriptors_per_figure = 15  # Number of descriptors per figure
cols = 5
rows = math.ceil(descriptors_per_figure / cols)  # Rows needed to accommodate the subplots

# Create a folder to save figures
output_folder = "boxplots"
os.makedirs(output_folder, exist_ok=True)

# Define the combinations of the second and third columns
combinations = [(0, 0), (1, 0), (0, 1), (1, 1)]
combination_labels = ['(0,0)', '(1,0)', '(0,1)', '(1,1)']

# Iterate through each group of descriptors
for fig_num in range(math.ceil(num_vars / descriptors_per_figure)):
    start_idx = fig_num * descriptors_per_figure  # Starting index for variables in this figure
    end_idx = min((fig_num + 1) * descriptors_per_figure, num_vars)  # Ending index for variables in this figure
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten()

    # Plot each descriptor
    for i, idx in enumerate(range(start_idx, end_idx)):
        col = df_scaled.columns[idx]
        
        combined_data = pd.DataFrame()
        for j, (val2, val3) in enumerate(combinations):
            filtered_data = df_scaled[(df.iloc[:, 1] == val2) & (df.iloc[:, 2] == val3)]
            temp_data = filtered_data[[col]].copy()
            temp_data['group'] = combination_labels[j]
            combined_data = pd.concat([combined_data, temp_data], axis=0)
        
        sns.boxplot(x='group', y=col, data=combined_data, ax=axes[i])
        axes[i].set_title(col, fontweight='bold', fontsize=10)
    
    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # Save the figure in the output folder
    fig.savefig(os.path.join(output_folder, f"boxplots_{fig_num + 1}.png"))
    plt.close(fig)  # Close the figure to free memory

