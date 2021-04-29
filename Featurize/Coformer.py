# -*- coding: utf-8 -*-
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from .CalcuDescriptors import CalcuDescriptors
from .AdjacentTensor import AdjacentTensor
from .Fingerprint import Fingerprint
from .VertexMatrix import VertexMatrix
from .Atom_Bond import Atom, Bond
import numpy as np
import math
import openbabel as ob
import pybel
import os


intervals = [(0,2), (2,2.5), (2.5,3), (3,3.5), (3.5,4),
             (4,4.5), (4.5,5), (5,5.5), (5.5,6), (6,float("inf"))]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x,type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def get_edges(rdkit_mol):
    edges = {}
    for b in rdkit_mol.GetBonds():
        start = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        edges.setdefault((start,end), Bond(b))
    return edges


class Coformer(object):
    def __init__(self, mol_file, removeh=False, fmt=None, smiles=False):
        self.removeh = removeh
        if fmt==None:
            fmt = mol_file.split('.')[-1]
        #####################################################################################
        if os.path.isfile(mol_file):
            self.ob_mol = pybel.readfile(fmt, mol_file).__next__()
            self.rdkit_mol = Chem.MolFromMolBlock(self.ob_mol.write('mol'), removeHs=False)
            self.ob_mol = self.ob_mol.OBMol
            self.molname = mol_file.split('/')[-1].split('.')[0]
        else:
            try:
                self.ob_mol = pybel.readstring(fmt, mol_file)
                self.rdkit_mol = Chem.MolFromMolBlock(self.ob_mol.write('mol'), removeHs=False)
                self.ob_mol = self.ob_mol.OBMol
                self.molname = mol_file.split('\n')[0].strip()
            except:
                self.ob_mol = pybel.readstring('smi', mol_file)
                self.ob_mol.make3D()
                self.rdkit_mol = Chem.MolFromMolBlock(self.ob_mol.write('mol'), removeHs=False)
                self.ob_mol = self.ob_mol.OBMol
                self.molname = None
        if self.rdkit_mol == None:
            raise ValueError('rdkit can not read:\n{}'.format(mol_file))
        ######################################################################################
        if self.removeh:
            self.ob_mol.DeleteHydrogens()
            self.rdkit_mol = Chem.RemoveHs(self.rdkit_mol)
        if len(self.rdkit_mol.GetAtoms()) != self.ob_mol.NumAtoms():
            raise ValueError('len(rdkit_mol.GetAtoms()) != len(ob_mol.atoms):\n{}'.format(mol_file))
            
        rd_atoms = {}
        ob_atoms = {}
        for i in range(self.ob_mol.NumAtoms()):
            rdcoord = tuple([round(j,4) for j in self.rdkit_mol.GetAtomWithIdx(i).GetOwningMol().GetConformer().GetAtomPosition(i)])
            rd_atoms.setdefault(rdcoord, self.rdkit_mol.GetAtomWithIdx(i))
            ob_atom = self.ob_mol.GetAtomById(i)
            obcoord = (round(ob_atom.GetX(),4), round(ob_atom.GetY(),4), round(ob_atom.GetZ(),4))
            ob_atoms.setdefault(obcoord, ob_atom)
        self.atoms = {}
        for key in rd_atoms:
            ix = rd_atoms[key].GetIdx()
            self.atoms.setdefault(ix, Atom(rd_atoms[key], ob_atoms[key]))
        self.atom_number = len(self.atoms)
        
    def descriptors(self, includeSandP=True, charge_model='eem2015bm'):
        return CalcuDescriptors(self, includeSandP=includeSandP, charge_model=charge_model)
    
    @property
    def AdjacentTensor(self):
        return AdjacentTensor(self.atoms, self.get_edges, self.atom_number)
    
    @property
    def Fingerprint(self):
        return Fingerprint(self.rdkit_mol)
    
    @property
    def VertexMatrix(self):
        return VertexMatrix(self.atoms)
    
    @property
    def get_edges(self):
        return get_edges(self.rdkit_mol)
    
    @property
    def hbond_donors(self):
        '''
        Get all H-bond donors
        '''
        hbond_donor_ix = {}
        for ix in self.atoms:
            if self.atoms[ix].feature.is_donor:
                Hs = self.atoms[ix].get_adjHs
                hbond_donor_ix.setdefault(ix, Hs)
        return hbond_donor_ix
    
    @property
    def hbond_acceptors(self):
        '''
        Get all H-bond acceptors
        '''
        self.hbond_acceptor_ix = []
        for ix in self.atoms:
            if self.atoms[ix].feature.is_acceptor:
                self.hbond_acceptor_ix.append(ix)
        return self.hbond_acceptor_ix
    
    @property
    def get_DHs(self):
        '''
        Get Donor-H bond
        '''
        DHs = []
        for D in self.hbond_donors:
            DHs = DHs + self.hbond_donors[D]
        return DHs
    
    @property
    def get_CHs(self):
        '''
        Get C-H bond
        '''
        CHs = []
        for ix in self.atoms:
            if self.atoms[ix].feature.symbol == 'C':
                Hs = self.atoms[ix].get_adjHs
                if len(Hs) != 0:
                    CHs = CHs + Hs
        return CHs
        
    @property
    def aromatic_atoms(self):
        '''
        Get all the aromatic atoms
        '''
        return [a.GetIdx() for a in self.rdkit_mol.GetAromaticAtoms()]