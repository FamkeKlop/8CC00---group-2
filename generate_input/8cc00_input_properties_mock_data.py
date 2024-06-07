import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Input defenition
infile = 'tested_molecules.csv'
df = pd.read_csv(infile)
smiles_column = 'SMILES'

# List properties
properties = {
    'MolecularWeight': Descriptors.MolWt,
    'LogP': Descriptors.MolLogP,
    'TPSA': rdMolDescriptors.CalcTPSA,
    'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds,
    'NumHDonors': rdMolDescriptors.CalcNumHBD,
    'NumHAcceptors': rdMolDescriptors.CalcNumHBA,
    'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings,
}

# Function to extract properties with SMILES as input
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {name: func(mol) for name, func in properties.items()}
    else:
        return {name: None for name in properties.keys()}
for prop_name in properties.keys():
    df[prop_name] = df[smiles_column].apply(lambda x: compute_properties(x).get(prop_name))

# Create new CSV file
outfile = 'tested_molecules_properties.csv'
df.to_csv(outfile, index=False)
