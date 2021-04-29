import sys
sys.path.append("..")
import numpy as np
import time
from Featurize import *
import glob
from multiprocessing import Pool


def GetNodeMask(graph_size, max_size=None):
    if max_size == None:
        max_size = np.max(graph_size)
    return np.array([np.pad(np.ones([s, 1]), ((0, max_size-s), (0, 0)), 'constant', constant_values=(0)) for s in graph_size], dtype=np.float32)


class Dataset(object):
    
    def __init__(self, table_dir, mol_blocks_dir='./Mol_Blocks.dir', fmt='sdf'):
        try:
            self.table = eval(open(table_dir).read())
        except:
            self.table = [i.strip().split('\t') for i in open(table_dir).readlines()]
        self.mol_blocks = eval(open(mol_blocks_dir).read())
        self.fmt = fmt
        
    def _task(self, items):
        array = {}
        tag = items[3]
        array.setdefault('tags',tag)
        block1 = self.mol_blocks[items[0]]
        block2 = self.mol_blocks[items[1]]
        try:
            c1 = Coformer(block1, fmt=self.fmt)
            c2 = Coformer(block2, fmt=self.fmt)
            cc = Cocrystal(c1, c2)
            label = int(items[2])
            array.setdefault('labels',label)
            subgraph_size = np.array([c1.atom_number, c2.atom_number])
            array.setdefault('subgraph_size', subgraph_size)
            if self.Desc:
                desc = cc.descriptors()
                array.setdefault('global_state',desc)
            if self.A_type:
                A = cc.CCGraphTensor(t_type=self.A_type, hbond=self.hbond, pipi_stack=self.pipi_stack, contact=self.contact)
                V = cc.VertexMatrix.feature_matrix() 
                array.setdefault('A', A)
                array.setdefault('V', V)
            if self.fp_type:
                fp = cc.Fingerprints(fp_type=self.fp_type, nBits=self.nBits)
                array.setdefault('fingerprints', fp)                                                                 
            return array
        except:
            print('Bad input sample:'+tag+', skipped.')
    
    def _PreprocessData(self, max_graph_size=None):
        graph_size = np.array([a.shape[0] for a in self.A]).astype(np.int32)
        if max_graph_size:
            largest_graph = max_graph_size
        else:
            largest_graph = max(graph_size)
        graph_vertices = []
        graph_adjacency = []
        for i in range(len(self.V)):
            # pad all vertices to match size
            graph_vertices.append(np.pad(self.V[i].astype(np.float32), 
                                             ((0, largest_graph-self.V[i].shape[0]), (0, 0)), 
                                             'constant', constant_values=(0)))
            # pad all adjacency matrices to match size
            new_A = self.A[i]
            graph_adjacency.append(np.pad(new_A.astype(np.float32), 
                                        ((0, largest_graph-new_A.shape[0]), (0, 0), (0, largest_graph-new_A.shape[0])), 
                                          'constant', constant_values=(0)))
        self.V = np.stack(graph_vertices, axis=0)
        self.A = np.stack(graph_adjacency, axis=0)
        self.labels = self.labels.astype(np.int32)
        if 'global_state' in self.__dict__:
            self.global_state = self.global_state.astype(np.float32)
        self.masks = GetNodeMask(graph_size, max_size=largest_graph)
        self.graph_size = graph_size
        self.subgraph_size = self.subgraph_size.astype(np.int32)

    def make_graph_dataset(self, Desc=0, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, 
                           processes=15, max_graph_size=None, make_dataframe=False, save_name=None):
        
        exception = {'BOVQUY','CEJPAK','GAWTON','GIPTAA','IDIGUY','LADBIB01','PIGXUY','SIFBIT','SOJZEW','TOFPOW',
                     'QOVZIK','RIJNEF','SIBFAK','SIBFEO','TOKGIJ','TOKGOP','TUQTEE','BEDZUF'}
        self.Desc = Desc
        self.A_type = A_type
        self.hbond = hbond
        self.pipi_stack = pipi_stack
        self.contact = contact
        self.fp_type = None
        start = time.time()
        pool = Pool(processes=processes)
        D = pool.map(self._task, [items for items in self.table if items[-1] not in exception])
        pool.close()
        pool.join()
        self.data_attr_names = D[0].keys()
        attrs = self.__dict__
        for k in self.data_attr_names:
            attrs.setdefault(k, [])
        for i in [j for j in D if j!=None]:
            for j in i:
                attrs[j].append(i[j])
        for l in attrs:
            if l in self.data_attr_names:
                attrs[l] = np.array(attrs[l])
        del D
        self._PreprocessData(max_graph_size=max_graph_size)
        if save_name:
            if 'global_state' in self.__dict__:
                np.savez(save_name, V=self.V, A=self.A, labels=self.labels, masks=self.masks, graph_size=self.graph_size, tags=self.tags, global_state=self.global_state, subgraph_size=self.subgraph_size)
            else:
                np.savez(save_name, V=self.V, A=self.A, labels=self.labels, masks=self.masks, graph_size=self.graph_size, tags=self.tags, subgraph_size=self.subgraph_size)
        if make_dataframe:
            self.dataframe = {}
            for ix,tag in enumerate(self.tags):
                self.dataframe.setdefault(tag, {})
                self.dataframe[tag].setdefault('V', self.V[ix])
                self.dataframe[tag].setdefault('A', self.A[ix])
                self.dataframe[tag].setdefault('label', self.labels[ix])
                if 'global_state' in self.__dict__:
                    self.dataframe[tag].setdefault('global_state', self.global_state[ix])
                self.dataframe[tag].setdefault('tag', tag)
                self.dataframe[tag].setdefault('mask', self.masks[ix])
                self.dataframe[tag].setdefault('graph_size', self.graph_size[ix])
                self.dataframe[tag].setdefault('subgraph_size', self.subgraph_size[ix])
            del self.V, self.A, self.labels, self.masks, self.graph_size, self.tags
            if 'global_state' in self.__dict__:
                del self.global_state
            if 'subgraph_size' in self.__dict__:
                del self.subgraph_size
            end = time.time()
            t = round(end-start, 2)
            print('Elapsed Time: '+str(t)+' s')
            return None
        end = time.time()
        t = round(end-start, 2)
        print('Elapsed Time: '+str(t)+' s')
        #return self
    
    def make_embedding_dataset(self, fp_type='ecfp', nBits=2048, processes=15, make_dataframe=True, save_name=None):
        exception = {'BOVQUY','CEJPAK','GAWTON','GIPTAA','IDIGUY','LADBIB01','PIGXUY','SIFBIT','SOJZEW','TOFPOW',
                     'QOVZIK','RIJNEF','SIBFAK','SIBFEO','TOKGIJ','TOKGOP','TUQTEE','BEDZUF'}
        self.fp_type = fp_type
        self.nBits = nBits
        self.Desc = 0
        self.A_type = 0
        start = time.time()
        pool = Pool(processes=processes)
        D = pool.map(self._task, [items for items in self.table if items[-1] not in exception])
        D = [i for i in D if i != None]
        pool.close()
        pool.join()
        self.data_attr_names = D[0].keys()
        attrs = self.__dict__
        for k in self.data_attr_names:
            attrs.setdefault(k, [])
        for i in D:
            for j in i:
                attrs[j].append(i[j])
        for l in attrs:
            if l in self.data_attr_names:
                attrs[l] = np.array(attrs[l])
        del D
        if save_name:
            np.savez(save_name, fingerprints=self.fingerprints, labels=self.labels, tags=self.tags)
        if make_dataframe:
            self.dataframe = {}
            for ix,tag in enumerate(self.tags):
                self.dataframe.setdefault(tag, {})
                self.dataframe[tag].setdefault('fingerprints', self.fingerprints[ix])
                self.dataframe[tag].setdefault('label', self.labels[ix])
                self.dataframe[tag].setdefault('tag', tag)
            del self.fingerprints, self.labels, self.tags
            end = time.time()
            t = round(end-start, 2)
            print('Elapsed Time: '+str(t)+' s')
            return self.dataframe
        end = time.time()
        t = round(end-start, 2)
        print('Elapsed Time: '+str(t)+' s')
        return self

    def _embedding_func(self, samples, dataframe):
        embedding, labels, tags = [], [], []
        for i in samples:
            embedding.append(dataframe[i]['fingerprints'])
            tags.append(dataframe[i]['tag'])
            labels.append(int(dataframe[i]['label']))
        data = [embedding, labels, tags]
        return np.array(embedding, dtype=np.float32), np.array(labels), np.array(tags)
        
    def _graph_func(self, samples, dataframe):
        V, A, labels, tags, desc, graph_size, masks, subgraph_size = [], [], [], [], [], [], [], []
        for i in samples:
            V.append(dataframe[i]['V'])
            A.append(dataframe[i]['A'])
            labels.append(int(dataframe[i]['label']))
            tags.append(dataframe[i]['tag'])
            graph_size.append(dataframe[i]['graph_size'])
            masks.append(dataframe[i]['mask'])
            subgraph_size.append(dataframe[i]['subgraph_size'])
            if self.Desc:
                desc.append(dataframe[i]['global_state'])
        if self.Desc:
            data = [V, A, labels, masks, graph_size, tags, desc, subgraph_size]
            return [np.array(i) for i in data]
        else:
            data = [V, A, labels, masks, graph_size, tags]
            return [np.array(i) for i in data]
            
    def split(self, train_samples=None, test_samples=None, val=False, val_samples=None, with_fps=False):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.val = val
        if self.val:
            self.val_samples = val_samples
        if with_fps:
            train_data = self._embedding_func(self.train_samples, self.dataframe)
            test_data = self._embedding_func(self.test_samples, self.dataframe)
            if self.val:
                val_data = self._embedding_func(self.val_samples, self.dataframe)
                return train_data, test_data, val_data
            else:
                return train_data, test_data
        else:
            train_data = self._graph_func(self.train_samples, self.dataframe)
            test_data = self._graph_func(self.test_samples, self.dataframe)
            if self.val:
                val_data = self._graph_func(self.val_samples, self.dataframe)
                return train_data, test_data, val_data
            else:
                return train_data, test_data


class DataLoader(Dataset):
    def __init__(self, npz_file, make_df=True):
        data = np.load(npz_file, allow_pickle=True)
        
        for key in data:
            self.__dict__[key] = data[key]
        del data
        if 'global_state' in self.__dict__:
            self.Desc = True
        if make_df:
            self.dataframe = {}
            for ix,tag in enumerate(self.tags):
                self.dataframe.setdefault(tag, {})
                self.dataframe[tag].setdefault('V', self.V[ix])
                self.dataframe[tag].setdefault('A', self.A[ix])
                self.dataframe[tag].setdefault('label', self.labels[ix])
                if 'global_state' in self.__dict__:
                    self.dataframe[tag].setdefault('global_state', self.global_state[ix])
                self.dataframe[tag].setdefault('tag', tag)
                self.dataframe[tag].setdefault('mask', self.masks[ix])
                self.dataframe[tag].setdefault('graph_size', self.graph_size[ix])
                if 'subgraph_size' in self.__dict__:
                    self.dataframe[tag].setdefault('subgraph_size', self.subgraph_size[ix])
            del self.V, self.A, self.labels, self.tags, self.masks, self.graph_size
            if 'global_state' in self.__dict__:
                del self.global_state
            if 'subgraph_size' in self.__dict__:
                del self.subgraph_size