# -*- coding: utf-8 -*-
import numpy as np
from rdkit import Chem
from rdkit.Chem import PeriodicTable

pt = Chem.GetPeriodicTable()

class atom_feat(object):
    def __init__(self, Atom):
        self.coordinates = Atom.ob_coor
        self.symbol = Atom.rdkit_atom.GetSymbol()
        self.hybridization = Atom.rdkit_atom.GetHybridization().__str__()
        self.chirality = Atom.rdkit_atom.GetChiralTag().__str__()  
        self.is_chiral = Atom.ob_atom.IsChiral()
        self.explicitvalence = Atom.rdkit_atom.GetExplicitValence()
        self.implicitvalence = Atom.rdkit_atom.GetImplicitValence()
        self.totalnumHs = Atom.rdkit_atom.GetTotalNumHs()
        self.formalcharge = Atom.rdkit_atom.GetFormalCharge()
        self.radical_electrons = Atom.rdkit_atom.GetNumRadicalElectrons()
        self.is_aromatic = Atom.rdkit_atom.GetIsAromatic()
        self.is_acceptor = Atom.ob_atom.IsHbondAcceptor()
        self.is_donor = Atom.ob_atom.IsHbondDonor()
        self.is_cyclic = Atom.rdkit_atom.IsInRing()
        self.is_metal = Atom.ob_atom.IsMetal
        self.atomic_weight = Atom.rdkit_atom.GetMass()
        self.atomic_number = Atom.rdkit_atom.GetAtomicNum()
        self.vdw_radius = pt.GetRvdw(self.symbol)
        self.sybyl_type = Atom.ob_atom.GetType()
        self.degree = Atom.rdkit_atom.GetDegree()
        
class Atom(object):
    def __init__(self, rdkit_atom, ob_atom):
        
        self.rdkit_atom, self.ob_atom = rdkit_atom, ob_atom
        self.idx = rdkit_atom.GetIdx()
        self.ob_coor = np.array([ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()])

    @property
    def feature(self):
        return atom_feat(self)
    
    @property
    def get_bonds(self):
        self.bonds = {}
        for bond in self.rdkit_atom.GetBonds():
            bond_key = (self.idx, bond.GetOtherAtomIdx(self.idx))
            self.bonds.setdefault(bond_key, Bond(bond))
        return self.bonds
    
    @property
    def get_adjHs(self):
        Hs = []
        for bond in self.rdkit_atom.GetBonds():
            adj_symbol = bond.GetOtherAtom(self.rdkit_atom).GetSymbol()
            if adj_symbol == 'H':
                Hs.append(bond.GetOtherAtomIdx(self.idx))
        return Hs

class Bond(object):
    def __init__(self, rdkit_bond):
        self.end_atom_idx = rdkit_bond.GetEndAtomIdx()
        self.begin_atom_idx = rdkit_bond.GetBeginAtomIdx()
        self.end_atom_coor = [i for i in rdkit_bond.GetOwningMol().GetConformer().GetAtomPosition(self.end_atom_idx)]
        self.begin_atom_coor = [i for i in rdkit_bond.GetOwningMol().GetConformer().GetAtomPosition(self.begin_atom_idx)]
        bond_type_list = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
        self.bond_type = rdkit_bond.GetBondType().__str__()
        try:
            self.type_number = bond_type_list.index(self.bond_type) + 1
        except:
            print('bond type is out of range {}'+self.bond_type)
            self.type_number = 5
        self.length = np.linalg.norm(np.array(self.end_atom_coor)-np.array(self.begin_atom_coor))
        self.is_ring = rdkit_bond.IsInRing()
        self.is_conjugated = rdkit_bond.GetIsConjugated()