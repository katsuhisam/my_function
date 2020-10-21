import pandas as pd
import numpy as np
from chem.fingerprints import hash2bits_pd
from chem.transform import SmilesToOEGraphMol
from chem.fingerprints import CalcECFPSparse
from functools import partial

def Smiles2Hash(smiles_series):
    """
    smiles to ECFP hash(2048bits)
    """
    
    CalcECFPSparse_2048 = partial(CalcECFPSparse, nbits=2048)
    
    mol = smiles_series.apply(SmilesToOEGraphMol)
    hsh = mol.apply(CalcECFPSparse_2048)
    
    return hsh


def Hash2FingerPrint(hash_series):
    """
    convert hash_series to ECFP4 2048bits
    """
    bits = hash2bits_pd(hash_series).replace({True:1, False:0})
    
    return bits


def Smiles2FingerPrint(smiles_series):
    """
    convert smiles to ECFP4 2048bits
    """
    hash_series = Smiles2Hash(smiles_series)
    bits = hash2bits_pd(hash_series).replace({True:1, False:0})
    
    return bits