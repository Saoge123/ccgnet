# -*- coding: utf-8 -*-
import numpy as np


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


element_symbol_list = ['Cl', 'N', 'P', 'Br', 'B', 'S', 'I', 'F', 'C', 'O', 'H']
possible_hybridization_types = ['SP2', 'SP3', 'SP', 'S','SP3D','SP3D2']
possible_chirality_list = ['', 'R', 'S']
possible_explicitvalence_list = [1, 2, 3, 4, 5, 6]
possible_implicitvalence_list =  [0, 1, 2, 3]
possible_numH_list = [0, 1, 2, 3]
possible_formal_charge_list = [0, 1, -1]
possible_radical_electrons_list = [0, 1]
is_aromatic = [False, True]
is_acceptor = [False, True]
is_donor = [False, True]
is_spiro = [False, True]
is_cyclic = [False, True]
is_metal = [False, True]
is_chiral = [False, True]
degree = [1, 2, 3, 4]

reference_dic = {'symbol':element_symbol_list, 
                 'hybridization':possible_hybridization_types,
                 'chirality':possible_chirality_list,
                 'is_chiral':is_chiral,
                 'explicitvalence':possible_explicitvalence_list,
                 'implicitvalence':possible_implicitvalence_list,
                 'totalnumHs':possible_numH_list,
                 'formalcharge':possible_formal_charge_list,
                 'radical_electrons':possible_radical_electrons_list,
                 'is_aromatic':is_aromatic,
                 'is_acceptor':is_acceptor,
                 'is_donor':is_donor,
                 'is_spiro':is_spiro,
                 'is_cyclic':is_cyclic,
                 'is_metal':is_metal,
                 'degree':degree}

'''
available_atom_feat = ['symbol','hybridization','explicitvalence','implicitvalence','totalnumHs',
                      'formalcharge','radical_electrons','is_aromatic','is_acceptor','is_donor','is_spiro',
                      'is_cyclic','is_metal','sybyl_type','atomic_weight','atomic_number','vdw_radius',
                      'degree','is_chiral','chirality']
'''
atom_feat_to_use = ['symbol','hybridization', 'chirality', 'is_chiral', 'is_spiro', 'is_cyclic',
                    'is_aromatic', 'is_donor', 'is_acceptor', 'degree','vdw_radius','explicitvalence',
                    'implicitvalence', 'totalnumHs', 'formalcharge','radical_electrons','atomic_number']

class VertexMatrix(object):
    def __init__(self, atoms):
        self.nodes = atoms
        self.node_number = len(atoms)
    
    def feature_matrix(self, atom_feat=atom_feat_to_use):
        V = []
        for node_ix in range(self.node_number):
            results = []
            node_features = self.nodes[node_ix].feature.__dict__
            results.extend(one_of_k_encoding(node_features['symbol'], reference_dic['symbol']))
            results.extend(one_of_k_encoding(node_features['hybridization'], reference_dic['hybridization']))
            results.extend(one_of_k_encoding(node_features['chirality'], reference_dic['chirality']))
            l = [float(node_features[feat_key]) for feat_key in atom_feat[3:]]
            results.extend(l)
            V.append(np.array(results, dtype='float32'))
        return np.array(V)