# -*- coding: utf-8 -*-
import numpy as np
from .CalcuDescriptors import CalcuDescriptors
from .AdjacentTensor import AdjacentTensor
from .Fingerprint import Fingerprint
from .VertexMatrix import VertexMatrix
from .Atom_Bond import Atom, Bond
from scipy.sparse import coo_matrix

def PadAdjMat(A1, A2):
    node_num1 = A1.shape[0]
    node_num2 = A2.shape[0]
    pad_A = []
    for i in range(A1.shape[1]):
        adj1_pad = np.pad(A1[:,i,:], ((0, 0), (0, node_num2)), 'constant', constant_values=(0))
        adj2_pad = np.pad(A2[:,i,:], ((0, 0), (node_num1, 0)), 'constant', constant_values=(0))
        pad_A.append(np.concatenate((adj1_pad, adj2_pad), axis=0))
    return np.array(Pad_A)

def combine(l1, l2):
    l = []
    if l1==[] or l2==[]:
        return l
    else:
        for i in l1:
            for j in l2:
                l.append([i, j])
        return l

def AdjMatOfInterMolecularInteraction(NodeNumber, interactions):
    interaction_mat = np.zeros([NodeNumber,1,NodeNumber])
    for act in interactions:
        interaction_mat[act[0],0,act[1]] = 1
        interaction_mat[act[1],0,act[0]] = 1
    return interaction_mat


class Cocrystal(object):
    
    def __init__(self, coformer1, coformer2):
        self.coformer1 = coformer1
        self.coformer2 = coformer2
        self.node_num1 = len(coformer1.atoms)
        self.node_num2 = len(coformer2.atoms)
        self.NodeNumber = self.node_num1+self.node_num2
        if coformer1.molname != None and coformer2.molname != None:
            self.name = coformer1.molname+'&'+coformer2.molname
        else:
            self.name = None
    
    @property
    def AdjacentTensor(self):
        return AdjacentTensor(self.get_nodes, self.get_edges, self.NodeNumber)
    
    @property
    def VertexMatrix(self):
        return VertexMatrix(self.get_nodes)
        
    @property
    def get_nodes(self):
        self.nodes = {}
        for atom_ix1 in self.coformer1.atoms:
            self.nodes.setdefault(atom_ix1, self.coformer1.atoms[atom_ix1])
        for atom_ix2 in self.coformer2.atoms:
            self.nodes.setdefault(atom_ix2+self.node_num1, self.coformer2.atoms[atom_ix2])
        return self.nodes
    
    @property
    def get_edges(self):
        edges = {}
        edges1 = self.coformer1.get_edges
        for k1 in edges1:
            edges.setdefault(k1, edges1[k1])
        edges2 = self.coformer2.get_edges
        for k2 in edges2:
            edges.setdefault((k2[0]+self.node_num1,k2[1]+self.node_num1), edges2[k2])
        return edges
    
    def descriptors(self, includeSandP=True, charge_model='eem2015bm'):
        desc1 = self.coformer1.descriptors(includeSandP=includeSandP, charge_model=charge_model)
        desc2 = self.coformer2.descriptors(includeSandP=includeSandP, charge_model=charge_model)
        return np.append(desc1, desc2, axis=0)
            
    def Fingerprints(self, fp_type='ecfp', **kwargs):
        '''
        fp_type should be one of  the fp set {'avalon', 'ecfp', 'maccs', 'rdkit'} and is case insensitive.
        '''
        fp1_ins = Fingerprint(self.coformer1.rdkit_mol)
        fp2_ins = Fingerprint(self.coformer2.rdkit_mol)
        FP_funcs = {}
        FP_funcs['avalon'] = [getattr(fp1_ins, 'AvalonFP'),getattr(fp2_ins, 'AvalonFP')]
        FP_funcs['ecfp'] = [getattr(fp1_ins, 'ECFP'),getattr(fp2_ins, 'ECFP')]
        FP_funcs['maccs'] = [getattr(fp1_ins, 'MACCSkeysFP'),getattr(fp2_ins, 'MACCSkeysFP')]
        FP_funcs['rdkit'] = [getattr(fp1_ins, 'RDKitFP'),getattr(fp2_ins, 'RDKitFP')]
        fp1 = FP_funcs[fp_type.lower()][0](**kwargs)
        fp2 = FP_funcs[fp_type.lower()][1](**kwargs)
        return np.append(fp1, fp2, axis=0)
    
    @property
    def possible_hbonds(self):
        # process data for graph-CNN
        d1 = self.coformer1.hbond_donors
        Hs1 = []
        for h1 in d1.values():
            Hs1 = Hs1 + h1
        a1 = self.coformer1.hbond_acceptors
        
        d2 = self.coformer2.hbond_donors
        Hs2 = []
        for h2 in d2.values():
            Hs2 = Hs2 + h2
        Hs2 = list(np.array(Hs2)+self.node_num1)
        a2 = list(np.array(self.coformer2.hbond_acceptors)+self.node_num1)
        return combine(Hs1,a2) + combine(Hs2,a1)
        
    @property
    def possible_interaction(self):
        CHs_1 = self.coformer1.get_CHs
        CHs_2 = self.coformer2.get_CHs
        CHs_2 = list(np.array(CHs_2)+self.node_num1)
        
        A_1 = self.coformer1.hbond_acceptors
        A_2 = list(np.array(self.coformer2.hbond_acceptors)+self.node_num1)
        
        C_H_A = combine(CHs_1, A_2) + combine(CHs_2, A_1)
        return C_H_A

    @property
    def possible_pipi_stack(self):
        aromatic1 = self.coformer1.aromatic_atoms
        aromatic2 = list(np.array(self.coformer2.aromatic_atoms)+self.node_num1)
        pipi_stack = combine(aromatic1,aromatic2)
        return pipi_stack

    def InteractionTensor(self, hbond=True, pipi_stack=False, contact=False):
        if hbond==False and pipi_stack==False and contact==False:
            return None
        interactions = []
        if hbond:
            hb_adj = AdjMatOfInterMolecularInteraction(self.NodeNumber, self.possible_hbonds)
            interactions.append(hb_adj)
        if pipi_stack:
            pipi_adj = AdjMatOfInterMolecularInteraction(self.NodeNumber, self.possible_pipi_stack)
            interactions.append(pipi_adj)
        if contact:
            contact_adj = AdjMatOfInterMolecularInteraction(self.NodeNumber, self.possible_interaction)
            interactions.append(contact_adj)
        return np.concatenate(interactions, axis=1)
    
    def CCGraphTensor(self, t_type='OnlyCovalentBond', hbond=True, pipi_stack=False, contact=False):
        t_type_set = {'allfeature','allfeaturebin','isringandconjugated','onlycovalentbond',
                      'withbinbistancematrix','withbondlenth','withdistancematrix'}
        t_type = t_type.lower()
        if t_type not in t_type_set:
            raise ValueError('t_type is case insensitive and should be one of the list :{}'.format(str(list(t_type_set))))
        CCG_ins = AdjacentTensor(self.get_nodes, self.get_edges, self.NodeNumber)
        methods = [i for i in dir(CCG_ins) if '__' not in i]
        methods_dict = {}
        for method in methods:
            methods_dict[method.lower()] = getattr(CCG_ins, method)
        CCG = methods_dict[t_type]()
        ten_interaction = self.InteractionTensor(hbond=hbond, pipi_stack=pipi_stack, contact=contact)
        if ten_interaction is None:
            return CCG
        else:
            return np.append(CCG, ten_interaction, axis=1)
    
    def COO_CCGraphTensor(self, **kwargs):
        CCG = self.CCGraphTensor(**kwargs)
        coo_CCG = coo_matrix(CCG.sum(axis=1))
        edge_index = [coo_CCG.row, coo_CCG.col]
        edge_attr = CCG[edge_index[0],:,edge_index[1]]
        return edge_index, edge_attr