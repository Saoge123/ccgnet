# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdFreeSASA
import openbabel as ob
import numpy as np
import math


'''
from rdkit.Chem import Lipinski, MolSurf, PeriodicTable
elements_set = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','Br','I']
pt = Chem.GetPeriodicTable()
Rvdw = {}
for i in elements_set:
    Rvdw[i] = pt.GetRvdw(i)
'''

Rvdw = {'H': 1.2,'He': 1.4,'Li': 1.82,'Be': 1.7,'B': 2.08,'C': 1.95,'N': 1.85,
        'O': 1.7,'F': 1.73,'Ne': 1.54,'Na': 2.27,'Mg': 1.73,'Al': 2.05,'Si': 2.1,
        'P': 2.08,'S': 2.0,'Cl': 1.97,'Ar': 1.88,'Br': 2.1,'I': 2.15}

"""
def coordinate_adjusting(rdkit_mol):
    atoms = rdkit_mol.GetAtoms()
    atom_coors = np.array([rdkit_mol.GetConformer().GetAtomPosition(i.GetIdx()) for i in atoms])
    wts = np.array([i.GetMass() for i in atoms])
    wts = np.expand_dims(wts,axis=1)
    x, y, z =  atom_coors[:,0],  atom_coors[:,1],  atom_coors[:,2]
    imt = [
             [(wts*(y**2+z**2)).sum(), (-1*wts*x*y).sum(), (-1*wts*x*z).sum()],
             [(-1*wts*y*x).sum(), (wts*(x**2+z**2)).sum(), (-1*wts*y*z).sum()],
             [(-1*wts*z*x).sum(), (-1*wts*z*y).sum(), (wts*(y**2+x**2)).sum()]
          ]
    imt = np.array(imt)
    eig_v,eig_m = np.linalg.eig(imt)
    am = atom_coors.dot(eig_m)
    #print(eig_m)
    return am
"""

def coordinate_adjusting(rdkit_mol):
    '''
    The algorithm refers from：http://sobereva.com/426
    '''
    mat_coor = [[0,1],[0,2],[1,2]]
    diag_coor = [[1,2],[0,2],[0,1]]
    atoms = rdkit_mol.GetAtoms()
    atom_coors = np.array([rdkit_mol.GetConformer().GetAtomPosition(i.GetIdx()) for i in atoms])
    wts = np.array([i.GetMass() for i in atoms])
    wts = np.expand_dims(wts,axis=1)
    diag_val = [np.sum(wts*atom_coors[:,i]**2) for i in diag_coor]
    mat_val = [np.sum(wts*np.prod(atom_coors[:,i],axis=1))*-1 for i in mat_coor]
    imt = np.zeros([3,3])
    for i in range(3):
        imt[i,i] = diag_val[i]
        pos = mat_coor[i]
        imt[pos[0],pos[1]] = mat_val[i]
        imt[pos[1],pos[0]] = mat_val[i]
    eig_v,eig_m = np.linalg.eig(imt)
    am = atom_coors.dot(eig_m)
    return am

def MaxMinValue(array):
    max_idx, max_val = np.argmax(array), np.max(array)
    min_idx, min_val = np.argmin(array), np.min(array)
    return int(max_idx), max_val, int(min_idx), min_val

def CalcuAxisLenth(rdkit_mol):
    coors = coordinate_adjusting(rdkit_mol)
    axis = []
    for i in range(3):
        max_idx, max_val, min_idx, min_val = MaxMinValue(coors[:,i])
        max_ = max_val + Rvdw[rdkit_mol.GetAtomWithIdx(max_idx).GetSymbol()]
        min_ = min_val - Rvdw[rdkit_mol.GetAtomWithIdx(min_idx).GetSymbol()]
        axis.append(max_-min_)
    S, M, L = sorted(axis)
    return S, M, L

def Ratio_S_M_L(rdkit_mol):
    S, M, L = CalcuAxisLenth(rdkit_mol)
    S_L = S/L
    M_L = M/L
    S_M = S/M
    return S_L, M_L, S_M, S

def GlobularityAndFrTPSA(rdkit_mol, includeSandP=1):
    # Calculate globularity: surface of a sphere with the same volume as the molecule / Area
    # FrTPSA = TPSA / SASA
    mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(mol)
    mol_vol = AllChem.ComputeMolVolume(mol,confId=0)
    r_sphere = math.pow(mol_vol*0.75/math.pi, 1.0/3)
    area_sphere = 4*math.pi*r_sphere*r_sphere
    radii = rdFreeSASA.classifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, radii)
    globularity = area_sphere / sasa
    FrTPSA = Descriptors.TPSA(mol, includeSandP=includeSandP) / sasa
    return globularity, FrTPSA

def FractionNO(rdkit_mol):
    return Descriptors.NOCount(rdkit_mol) / float(rdkit_mol.GetNumHeavyAtoms())

def FractionAromaticAtoms(rdkit_mol):
    return len(rdkit_mol.GetAromaticAtoms()) / float(rdkit_mol.GetNumHeavyAtoms())

def NumHAcceptorsAndDonors(rdkit_mol):
    return Descriptors.NumHDonors(rdkit_mol), Descriptors.NumHAcceptors(rdkit_mol)

def RotatableBondNumber(rdkit_mol):
    mol = Chem.RemoveHs(rdkit_mol)  
    return Descriptors.NumRotatableBonds(rdkit_mol) #/ float(mol.GetNumBonds())

# Using openbabel for calculating dipole moment
def DipoleMoment(ob_mol, charge_model='eem2015bm'):
    #mol_block = Chem.MolToMolBlock(mol)
    #ob_mol = pybel.readstring('mol', mol_block)
    # We choose 'eem2015bm' to calculate dipole
    # Using 'obabel -L charges' can get a list of charge models
    dipole = ob.OBChargeModel_FindType(charge_model).GetDipoleMoment(ob_mol)
    dipole_moment = math.sqrt(dipole.GetX()**2+dipole.GetY()**2+dipole.GetZ()**2)
    return dipole_moment

def CalcuDescriptors(mol, includeSandP=1, charge_model='eem2015bm'):
    S_L, M_L, S_M, S = Ratio_S_M_L(mol.rdkit_mol)
    globularity, FrTPSA = GlobularityAndFrTPSA(mol.rdkit_mol, includeSandP=includeSandP)
    Fr_NO = FractionNO(mol.rdkit_mol)
    Fr_AromaticAtoms = FractionAromaticAtoms(mol.rdkit_mol)
    HBA,HBD = NumHAcceptorsAndDonors(mol.rdkit_mol)
    RBN = RotatableBondNumber(mol.rdkit_mol)
    dipole_moment = DipoleMoment(mol.ob_mol, charge_model=charge_model)
    return np.array([S_L, M_L, S_M, S, globularity, FrTPSA, Fr_NO, Fr_AromaticAtoms, HBA, HBD, RBN, dipole_moment])