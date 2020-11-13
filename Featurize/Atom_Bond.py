# -*- coding: utf-8 -*-
import ccdc
import numpy as np

HBondCriterion = ccdc.molecule.Molecule.HBondCriterion()

class atom_feat(object):
    def __init__(self, Atom, hb_criterion=HBondCriterion):
        self.coordinates = Atom.rdkit_coor
        self.symbol = Atom.rdkit_atom.GetSymbol()   # 元素符号
        self.hybridization = Atom.rdkit_atom.GetHybridization().__str__()  # 杂化类型， 关于 R-O-CO-R, RCOOH中O的杂化类型，rdkit给出的是sp2，ccdc给出的是sp3
        #chirality = atom.GetChiralTag().__str__()    # 手性
        self.chirality = Atom.csd_atom.chirality    # 手性，Returns one of '', 'R', 'S', 'Mixed', 'Error'
        self.is_chiral = Atom.csd_atom.is_chiral       # 是否为手性原子
        self.explicitvalence = Atom.rdkit_atom.GetExplicitValence()  # 化合价
        self.implicitvalence = Atom.rdkit_atom.GetImplicitValence()  # 隐式化合价
        self.totalnumHs = Atom.rdkit_atom.GetTotalNumHs()   # 与该原子相连的氢原子数
        #print(totalnumHs)
        self.formalcharge = Atom.rdkit_atom.GetFormalCharge()  # 形式电荷
        self.radical_electrons = Atom.rdkit_atom.GetNumRadicalElectrons()  #自由基电子数
        self.is_aromatic = Atom.rdkit_atom.GetIsAromatic()     # 是否芳香原子
        
        self.is_acceptor = hb_criterion.is_acceptor(Atom.csd_atom)   # 是否为氢键受体
        self.is_donor = hb_criterion.is_donor(Atom.csd_atom)         # 是否为氢键供体
        self.is_spiro = Atom.csd_atom.is_spiro         # 是否是螺原子
        self.is_cyclic = Atom.csd_atom.is_cyclic       # 是否在环上
        self.is_metal = Atom.csd_atom.is_metal         # 是否是金属

        self.atomic_weight = Atom.rdkit_atom.GetMass()   # 相对原子质量
        self.atomic_number = Atom.rdkit_atom.GetAtomicNum()  # 元素序号（元素周期表中）
        self.vdw_radius = Atom.csd_atom.vdw_radius     # 范德华半径
        self.sybyl_type = Atom.csd_atom.sybyl_type     # sybyl原子类型，参见：http://www.tri-ibiotech.com.cn/Appofcase/n247.html
        self.degree = Atom.rdkit_atom.GetDegree()         # 度
        
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
        self.length = np.linalg.norm(np.array(self.end_atom_coor)-np.array(self.begin_atom_coor))  # 键长
        self.is_ring = rdkit_bond.IsInRing()  # 键是否是环的一部分
        self.is_conjugated = rdkit_bond.GetIsConjugated()  # 是否共轭