import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from chem.transform import SmilesToOEGraphMol, MakeMolsFromPDdataFrame
from chem.readWrite import WriteMolsToSDF
from openeye.oechem import *

def GenSDFWithID(df, outfilepath):
    
    """
    SDF file makeing from chemblx24 data
    """
    
    mols = []
    for idx in tqdm(df.index):
        
        smi = df.loc[idx, "non_stereo_aromatic_smieles"]
        mol = SmilesToOEGraphMol(smi)

        OESetSDData(mol, "ID", str(df.loc[idx, "funatsu_lab_id"]))

        mols.append(mol)

    WriteMolsToSDF(mols, outfilepath)
    
def GenSDFWithIDSmiles(df, outfilepath):
    
    """
    SDF file making from smiles
    """
    
    mols = []
    for idx in tqdm(df.index):
        
        smi = df.loc[idx, "Washed_Smiles"] #or Washed_Smiles
        mol = SmilesToOEGraphMol(smi)

        OESetSDData(mol, "Molecule ChEMBL ID", str(df.loc[idx, "Molecule ChEMBL ID"]))

        mols.append(mol)

    WriteMolsToSDF(mols, outfilepath) 


if __name__ == "__main__":
    
    os.chdir("Z:/")
    df = pd.read_csv("./test_origin.txt", index_col=None, sep="\t")
    
    GenSDFWithID(df, "./test.sdf")