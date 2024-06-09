import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator



# Input and output definition
infile = 'tested_molecules.csv'
outfile = 'fingerprint_df.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(infile)
smiles_column = 'SMILES'

# Get actual fingerprints from all molecules

# Morgan fingerprints:
# fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
# RDKIt
# fpgen = AllChem.GetRDKitFPGenerator()
# ECPF/Morgan? For first SMILE
#fpgen1024 =  AllChem.GetMorganFingerprintAsBitVect()
#fpgen2048 = 

#df['fingerprints_1024'] = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius = 3, nBits = 2**10, useFeatures = False, useChirality = False) for smiles in df[smiles_column]]
#df['fingerprints_2048'] = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius = 3, nBits = 2**12, useFeatures = False, useChirality = False) for smiles in df[smiles_column]]

# Number of bits to present in binary vector:
N = [2**10, 2**11]

for n in N:
    bit_objects = []
    bit_lists = []
    for smiles in df[smiles_column]:
        
        # Get the bit object vector
        bit_object = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 
                                              radius = 3, 
                                              nBits = n, 
                                              useFeatures = False, 
                                              useChirality = False)
        bit_objects.append(bit_object)
        # Convert to a list for actual binary numbers
        bit_list = np.array(bit_object).tolist()
        bit_lists.append(bit_list)
        
    # Put both in df
    df[f'fingerprints_object_{n}'] = bit_objects
    df[f'fingerprints_list_{n}'] = bit_lists
                                              
# Write output
df.to_csv(outfile, index=False)

print("done")

