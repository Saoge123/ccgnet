# -*- coding: utf-8 -*-
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Fingerprints import FingerprintMols

class Fingerprint(object):

    def __init__(self, mol):
        self.mol = mol
        
    def ECFP(self, radii=2, nBits=2048):
        '''
        ECFP4: radii=2; ECFP6: radii=3; ECFP8: radii=4; 
        the rest can be done in the same manner.
        '''
        fps = AllChem.GetMorganFingerprintAsBitVect(self.mol, radii, nBits=nBits)
        return np.array(fps)

    def MACCSkeysFP(self):
        return np.array(MACCSkeys.GenMACCSKeys(self.mol))

    def AvalonFP(self, nBits=1024):
        return np.array(pyAvalonTools.GetAvalonFP(self.mol, nBits=nBits))

    def RDKitFP(self):
        return np.array(FingerprintMols.GetRDKFingerprint(self.mol))