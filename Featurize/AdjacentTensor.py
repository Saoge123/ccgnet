# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix

def bins(value):
    intervals = [(0,2), (2,2.5), (2.5,3), (3,3.5), (3.5,4),
                 (4,4.5), (4.5,5), (5,5.5), (5.5,6), (6,float("inf"))]
    for ix,b in enumerate(intervals):
        if b[0]<=value<b[1]:
            return ix

def bin_distance(atoms):
    '''
    embedding the distance as a one-hot vector, and return a tensor with shape=(N, 10, N)
    '''
    atom_number = len(atoms)
    bin_A = np.zeros([atom_number, 10, atom_number])
    for i in range(atom_number):
        coor1 = np.array(atoms[i].feature.coordinates)
        for j in range(i+1, atom_number):
            coor2 = np.array(atoms[j].feature.coordinates)
            distance = np.linalg.norm(coor1-coor2)
            b = bins(distance)
            bin_A[i,b,j] = 1
            bin_A[j,b,i] = 1
    return bin_A


class AdjacentTensor(object):
    def __init__(self, atoms, edges, atom_number):
        self.atom_number = atom_number
        self.atoms = atoms
        self.edges = edges
        
    def OnlyCovalentBond(self):
        A = np.zeros([self.atom_number,4,self.atom_number])
        for e in self.edges:
            e_features = self.edges[e]
            e_type = e_features.type_number
            A[e[0],e_type-1,e[1]] = 1
            A[e[1],e_type-1,e[0]] = 1
        return A

    def WithBondLenth(self):
        A = np.zeros([self.atom_number,5,self.atom_number])
        for e in self.edges:
            e_features = self.edges[e]
            e_type = e_features.type_number
            A[e[0],e_type-1,e[1]] = 1
            A[e[1],e_type-1,e[0]] = 1
            length = e_features.length
            A[e[0],4,e[1]] = length
            A[e[1],4,e[0]] = length
        return A

    def WithDistanceMatrix(self):
        coordinates_mat = np.array([self.atoms[i].feature.coordinates for i in self.atoms.keys()])
        distance_matrix = np.expand_dims(squareform(pdist(coordinates_mat,metric='euclidean')),1)
        A = np.zeros([self.atom_number,4,self.atom_number])
        for e in self.edges:
            edge_features = self.edges[e]
            e_type = edge_features.type_number
            A[e[0],e_type-1,e[1]] = 1
            A[e[1],e_type-1,e[0]] = 1
        return np.concatenate([A, distance_matrix], axis=1)

    def WithRingAndConjugated(self):
        A = np.zeros([self.atom_number,6,self.atom_number])
        for e in self.edges:
            edge_features = self.edges[e]
            e_type = edge_features.type_number
            A[e[0],e_type-1,e[1]] = 1
            A[e[1],e_type-1,e[0]] = 1
            A[e[0],4,e[1]] = int(edge_features.is_conjugated)
            A[e[1],4,e[0]] = int(edge_features.is_conjugated)
            A[e[0],5,e[1]] = int(edge_features.is_ring)
            A[e[1],5,e[0]] = int(edge_features.is_ring)
        return A

    def AllFeature(self):
        coordinates_mat = np.array([self.atoms[i].feature.coordinates for i in self.atoms.keys()])
        distance_matrix = np.expand_dims(squareform(pdist(coordinates_mat,metric='euclidean')),1)
        A = np.zeros([self.atom_number,6,self.atom_number])
        for e in self.edges:
            edge_features = self.edges[e]
            e_type = edge_features.type_number
            A[e[0],e_type-1,e[1]] = 1
            A[e[1],e_type-1,e[0]] = 1
            A[e[0],4,e[1]] = int(edge_features.is_conjugated)
            A[e[1],4,e[0]] = int(edge_features.is_conjugated)
            A[e[0],5,e[1]] = int(edge_features.is_ring)
            A[e[1],5,e[0]] = int(edge_features.is_ring)
        return np.concatenate([A, distance_matrix], axis=1)

    def WithBinDistanceMatrix(self):
        BinDistanceMatrix = bin_distance(self.atoms)
        A = np.zeros([self.atom_number, 4, self.atom_number])
        for e in self.edges:
            e_features = self.edges[e]
            e_type = e_features.type_number
            A[e[0],e_type-1,e[1]] = 1
            A[e[1],e_type-1,e[0]] = 1
        return np.concatenate([A, BinDistanceMatrix], axis=1)

    def AllFeatureBin(self):
        BinDistanceMatrix = bin_distance(self.atoms)
        A = np.zeros([self.atom_number, 6, self.atom_number])
        for e in self.edges:
            edge_features = self.edges[e]
            e_type = edge_features.type_number
            A[e[0],e_type-1,e[1]] = 1
            A[e[1],e_type-1,e[0]] = 1
            A[e[0],4,e[1]] = int(edge_features.is_conjugated)
            A[e[1],4,e[0]] = int(edge_features.is_conjugated)
            A[e[0],5,e[1]] = int(edge_features.is_ring)
            A[e[1],5,e[0]] = int(edge_features.is_ring)
        return np.concatenate([A, BinDistanceMatrix], axis=1)