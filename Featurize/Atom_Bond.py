# -*- coding: utf-8 -*-
import ccdc
import numpy as np

HBondCriterion = ccdc.molecule.Molecule.HBondCriterion()

class atom_feat(object):
    def __init__(self, Atom, hb_criterion=HBondCriterion):
        self.coordinates = Atom.rdkit_coor
        self.symbol = Atom.rdkit_atom.GetSymbol()
        self.hybridization = Atom.rdkit_atom.GetHybridization().__str__()
        
        self.chirality = Atom.csd_atom.chirality
        self.is_chiral = Atom.csd_atom.is_chiral
        self.explicitvalence = Atom.rdkit_atom.GetExplicitValence()
        self.implicitvalence = Atom.rdkit_atom.GetImplicitValence()
        self.totalnumHs = Atom.rdkit_atom.GetTotalNumHs()
        
        self.formalcharge = Atom.rdkit_atom.GetFormalCharge()
        self.radical_electrons = Atom.rdkit_atom.GetNumRadicalElectrons()
        self.is_aromatic = Atom.rdkit_atom.GetIsAromatic()
        
        self.is_acceptor = hb_criterion.is_acceptor(Atom.csd_atom)
        self.is_donor = hb_criterion.is_donor(Atom.csd_atom)
        self.is_spiro = Atom.csd_atom.is_spiro
        self.is_cyclic = Atom.csd_atom.is_cyclic
        self.is_metal = Atom.csd_atom.is_metal

        self.atomic_weight = Atom.rdkit_atom.GetMass()
        self.atomic_number = Atom.rdkit_atom.GetAtomicNum()
        self.vdw_radius = Atom.csd_atom.vdw_radius
        self.sybyl_type = Atom.csd_atom.sybyl_type
        self.degree = Atom.rdkit_atom.GetDegree()
        
class Atom(object):
    def __init__(self, rdkit_atom, csd_atom, hb_criterion=HBondCriterion):
        
        self.rdkit_atom, self.csd_atom = rdkit_atom, csd_atom
        self.hb_criterion = hb_criterion
        self.idx = rdkit_atom.GetIdx()
        self.rdkit_coor = np.array([i for i in rdkit_atom.GetOwningMol().GetConformer().GetAtomPosition(self.idx)])
        self.index = csd_atom.index
        self.csd_coor = np.array([i for i in csd_atom.coordinates])
        if False in list(self.rdkit_coor == self.csd_coor):
            raise ValueError('the index of csd_mol and rdkit_mol is different !')

    @property
    def feature(self):
        return atom_feat(self, hb_criterion=self.hb_criterion)
    
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