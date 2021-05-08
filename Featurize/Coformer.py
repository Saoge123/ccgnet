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
import ccdc
from ccdc import io
import os



HBondCriterion = ccdc.molecule.Molecule.HBondCriterion()

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
    def __init__(self, mol_file, removeh=False, hb_criterion=HBondCriterion):
        self.removeh = removeh
        #####################################################################################
        try:
            self.csd_mol = io.MoleculeReader(mol_file)[0]
        except:
            raise ValueError('ccdc canot read this file: {}'.format(mol_file))
        informat = mol_file.split('.')[-1]
        self.csd_mol = self.csd_mol.components[0]
        self.csd_atoms = self.csd_mol.atoms
        ######################################################################################
        if os.path.isfile(mol_file):
            if '.sdf' in mol_file or '.mol' in mol_file:
                self.rdkit_mol = AllChem.MolFromMolFile(mol_file, removeHs=False)
            elif '.mol2' in mol_file:
                self.rdkit_mol = AllChem.MolFromMol2File(mol_file, removeHs=False)
            self.molname = mol_file.split('/')[-1].split('.')[0]
        else:
            self.rdkit_mol = AllChem.MolFromMolBlock(mol_file, removeHs=False)
            self.molname = mol_file.split('\n')[0].strip()
            
        if self.rdkit_mol == None:
            csd_mol_ = io.MoleculeReader(mol_file)[0].components[0]
            self.rdkit_mol = AllChem.MolFromMol2Block(csd_mol_.to_string('mol2'), removeHs=False)
            if self.rdkit_mol == None:
                raise ValueError('rdkit can not read this file:{}'.format(mol_file))
        
        if len(self.rdkit_mol.GetAtoms()) != len(self.csd_mol.atoms):
            print('len(rdkit_mol.GetAtoms()) != len(csd_mol.atoms):{}'.format(mol_file))
            self.rdkit_mol = AllChem.MolFromMolBlock(self.csd_mol.to_string('sdf'), removeHs=False)
            print(len(self.rdkit_mol.GetAtoms()), len(self.csd_mol.atoms))
            
        if self.removeh:
            self.csd_mol.remove_hydrogens()
            self.rdkit_mol = Chem.RemoveHs(self.rdkit_mol)
            self.csd_atoms = self.csd_mol.atoms
        
        self.atoms = {}
        for ix, atom in enumerate(self.rdkit_mol.GetAtoms()):
            self.atoms.setdefault(ix, Atom(atom, self.csd_atoms[ix], hb_criterion=hb_criterion))
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