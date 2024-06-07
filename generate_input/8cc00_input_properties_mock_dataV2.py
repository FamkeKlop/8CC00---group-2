import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Input definition
infile = 'tested_molecules.csv'
df = pd.read_csv(infile)
smiles_column = 'SMILES'

# Get all available descriptors
all_descriptors = Descriptors.descList

# Create a dictionary for descriptor names and their corresponding functions
properties = {name: func for name, func in all_descriptors}

# Function to extract properties with SMILES as input
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {name: func(mol) for name, func in properties.items()}
    else:
        return {name: None for name in properties.keys()}

# Compute and add each property to the DataFrame
for prop_name in properties.keys():
    df[prop_name] = df[smiles_column].apply(lambda x: compute_properties(x).get(prop_name))

# Create new CSV file
outfile = 'tested_molecules_properties_all.csv'
df.to_csv(outfile, index=False)

# Print all extracted property names
print("Extracted Properties:")
for prop_name in properties.keys():
    print(prop_name)
