import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, Dropout, BatchNorm1d
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader as PyGDataloader
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar, Data, Dataset, DataLoader, DataListLoader)
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, balanced_accuracy_score, f1_score, r2_score
from ccgnet.Dataset import Dataset, DataLoader
import time
import random


class NN(torch.nn.Module):
    def __init__(self, 
                 dim, 
                 edge_feat_num,
                 act_fun,
                 lin1_size,
                 lin2_size):
        super(NN, self).__init__()
        self.dim = dim
        self.edge_feat_num = edge_feat_num
        self.act_fun = act_fun
        self.lin1_size = lin1_size
        self.lin2_size = lin2_size
        
        if self.lin1_size != None:
            self.lin1 = Linear(edge_feat_num, self.lin1_size)
            self.bn1 = torch.nn.BatchNorm1d(self.lin1_size)
            
        if self.lin2_size != None:
            if self.lin1_size == None:
                in_dim = self.edge_feat_num
            else:
                in_dim = self.lin1_size
            self.lin2 = Linear(in_dim, self.lin2_size)
            self.bn2 = torch.nn.BatchNorm1d(self.lin2_size)
            
        if self.lin1_size == None and self.lin2_size == None:
            self.lin3 = Linear(edge_feat_num, self.dim*self.dim)
            self.bn3 = torch.nn.BatchNorm1d(self.dim*self.dim)
        elif self.lin1_size == None and self.lin2_size != None:
            self.lin3 = Linear(self.lin2_size, self.dim*self.dim)
            self.bn3 = torch.nn.BatchNorm1d(self.dim*self.dim)
        elif self.lin1_size != None and self.lin2_size == None:
            self.lin3 = Linear(self.lin1_size, self.dim*self.dim)
            self.bn3 = torch.nn.BatchNorm1d(self.dim*self.dim)
        else:
            self.lin3 = Linear(self.lin2_size, self.dim*self.dim)
            self.bn3 = torch.nn.BatchNorm1d(self.dim*self.dim)
            
    def forward(self, x):
        if self.lin1_size != None:
            x = self.lin1(x)
            x = self.bn1(x)
            x = self.act_fun(x)
        if self.lin2_size != None:
            x = self.lin2(x)
            x = self.bn2(x)
            x = self.act_fun(x)
        x = self.lin3(x)
        x = self.bn3(x)
        return x

def build_model(dim,
                mp_step,
                act_fun,
                lin1_size,
                lin2_size,
                processing_steps,
                readout_act_func,
                readout_p_dropout,
                readout_dense_1,
                readout_dense_2,
                readout_dense_3
                ):
    class Net(torch.nn.Module):
        def __init__(self, dim=64, edge_feat_num=4, node_feat_num=34):
            super(Net, self).__init__()
            self.lin0 = torch.nn.Linear(node_feat_num, dim)
            self.e_nn = NN(dim, 
                           edge_feat_num,
                           act_fun,
                           lin1_size,
                           lin2_size)
            self.conv = NNConv(dim, dim, self.e_nn, aggr='mean', root_weight=False)
            self.gru = GRU(dim, dim)
            self.set2set = Set2Set(dim, processing_steps=processing_steps)
            
            self.readout_lin1 = Linear(dim*2, readout_dense_1)
            self.readout_bn1 = torch.nn.BatchNorm1d(readout_dense_1)
            if readout_dense_2 != None:
                self.readout_lin2 = Linear(readout_dense_1, readout_dense_2)
                self.readout_bn2 = torch.nn.BatchNorm1d(readout_dense_2)
            if readout_dense_3 != None:
                if readout_dense_2 == None:
                    in_dim = readout_dense_1
                else:
                    in_dim = readout_dense_2
                self.readout_lin3 = Linear(in_dim, readout_dense_3)
                self.readout_bn3 = torch.nn.BatchNorm1d(readout_dense_3)
            
            if readout_dense_3 == None and readout_dense_2 == None:
                self.readout_lin_out = Linear(readout_dense_1, 2)
            elif readout_dense_3 == None and readout_dense_2 != None:
                self.readout_lin_out = Linear(readout_dense_2, 2)
            else:
                self.readout_lin_out = Linear(readout_dense_3, 2)
            

        def forward(self, data):
            out = F.relu(self.lin0(data.x))
            h = out.unsqueeze(0)
            for i in range(mp_step):
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
            
            # Readout
            out = self.set2set(out, data.batch)
            # Dense Layers
            ## Dense Layer 1
            out = self.readout_lin1(out)
            out = self.readout_bn1(out)
            out = readout_act_func(out)
            out = F.dropout(out, p=readout_p_dropout)
            ## Dense Layer 2
            if readout_dense_2 != None:
                out = self.readout_lin2(out)
                out = self.readout_bn2(out)
                out = readout_act_func(out)
                out = F.dropout(out, p=readout_p_dropout)
            ## Dense Layer 3
            if readout_dense_3 != None:
                out = self.readout_lin3(out)
                out = self.readout_bn3(out)
                out = readout_act_func(out)
                out = F.dropout(out, p=readout_p_dropout)
            
            out = self.readout_lin_out(out)
            return F.log_softmax(out, dim=-1)
    net = Net(dim=dim, edge_feat_num=4, node_feat_num=34)
    return net


def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def coo_format(A):
    coo_A = np.zeros([A.shape[0],A.shape[2]])
    for i in range(A.shape[1]):
        coo_A = coo_A + A[:,i,:]
    coo_A = coo_matrix(coo_A)
    edge_index = [coo_A.row, coo_A.col]
    edge_attr = []
    for j in range(len(edge_index[0])):
        edge_attr.append(A[edge_index[0][j],:,edge_index[1][j]])
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr

def func(pred):
    return pred.index(max(pred))

def train(model, train_loader, device, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all/len(train_loader.dataset)

def test(model, loader, device):
    model.eval()
    correct = 0
    loss_all = 0
    model_output = []
    y = []
    tags = []
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        loss = F.nll_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs  
        correct += pred.eq(data.y).sum().item()
        model_output.extend(output.tolist())
        y.extend(data.y.tolist())
        tags.extend(data.tag)
    return float(correct)/len(loader.dataset), loss_all/len(loader.dataset), model_output, y, tags


from torch_geometric.data import DataLoader as PyGDataloader

class GetInputData(object):
    def __init__(self, dataframe):
        df = dataframe
        self.graphs = {}
        for i in df:
            x = torch.tensor(df[i]['V'][:df[i]['graph_size']])
            edge_index, edge_attr = coo_format(df[i]['A'])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            y = torch.tensor(np.array([df[i]['label']]), dtype=torch.long)
            data = Data(x=x, 
                        y=y,
                        edge_attr=edge_attr, 
                        edge_index=edge_index,
                        tag=i)
            self.graphs.setdefault(i, data)
            
    def split(self, train_samples=None, valid_samples=None, batch_size=128, with_test=False, test_samples=None):
        train_list = []
        valid_list = []
        if with_test:
            test_list = []
        for i in train_samples:
            train_list.append(self.graphs[i])
        for i in valid_samples:
            valid_list.append(self.graphs[i])
        if with_test:
            for i in test_samples:
                test_list.append(self.graphs[i])
        
        if with_test:
            return train_list, valid_list, test_list
        else:
            return train_list, valid_list

def make_dataset():
    data1 = Dataset('CC_Table/CC_Table.tab', mol_blocks_dir='Mol_Blocks.dir')
    data1.make_graph_dataset(Desc=0, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
    return data1

def dropout_rate():
    n = 0.0
    l = []
    for i in range(601):
        l.append(n)
        n += 0.001
    return l

def black_box_function(args_dict):
    batch_size = args_dict['batch_size']
    dim = args_dict['dim']
    mp_step = args_dict['mp_step']
    act_fun = args_dict['act_fun']
    lin1_size = args_dict['lin1_size']
    lin2_size = args_dict['lin2_size']
    processing_steps = args_dict['processing_steps']
    readout_act_func = args_dict['readout_act_func']
    readout_p_dropout = args_dict['readout_p_dropout']
    readout_dense_1 = args_dict['readout_dense_1']
    readout_dense_2 = args_dict['readout_dense_2']
    readout_dense_3 = args_dict['readout_dense_3']
    print(str(args_dict))
    
    train_loader = PyGDataloader(train_list, batch_size=batch_size, shuffle=True)
    valid_loader = PyGDataloader(valid_list, batch_size=batch_size, shuffle=False)
    # make save dir
    snapshot_path = abs_path+'/bayes_snapshot/'
    model_name = 'BayesOpt-MPNN/'
    verify_dir_exists(snapshot_path+model_name)
    if os.listdir(snapshot_path+model_name) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(snapshot_path+model_name) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_)+1)
    # build model
    MPNN = build_model(dim, 
                       mp_step,
                       act_fun,
                       lin1_size,
                       lin2_size,
                       processing_steps,
                       readout_act_func,
                       readout_p_dropout,
                       readout_dense_1,
                       readout_dense_2,
                       readout_dense_3
                       )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPNN.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=100, min_lr=0.00001)
    history = {}
    history['Train Loss'] = []
    history['Train Acc'] = []
    history['Valid Acc'] = []
    history['Valid Loss'] = []

    path = abs_path+'/bayes_snapshot'+'/'+model_name+'/'+dataset_name+'/'+'time_0/'
    reports = {}
    reports['valid acc'] = 0.0
    reports['valid loss'] = float('inf')
    for epoch in range(1, 101):
        start_time_1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(model, train_loader, device, optimizer)
        train_acc, _, _, _, _ = test(model, train_loader, device)
        valid_acc, valid_loss, valid_output, y_valid, valid_tags = test(model, valid_loader, device)
        scheduler.step(valid_acc)
        history['Train Loss'].append(train_loss)
        history['Train Acc'].append(train_acc)
        history['Valid Acc'].append(valid_acc)
        history['Valid Loss'].append(valid_loss)
        if valid_loss < reports['valid loss']:
            verify_dir_exists(path)
            torch.save(model.state_dict(), path+'/ModelParams.pkl')
            open(path+'/model-{}_info.txt'.format(epoch), 'w').write('\n'.join(['Step:{}'.format(epoch), 
                                                                                'valid_acc:{}'.format(valid_acc),
                                                                                'valid_cross_entropy:{}'.format(valid_loss),
                                                                                'train_acc:{}'.format(train_acc),
                                                                                'train_cross_entropy:{}'.format(train_loss), 
                                                                                str(y_valid), 
                                                                                str(valid_output), 
                                                                                str(valid_tags)]))
            reports['valid acc'] = valid_acc
            reports['valid loss'] = valid_loss
        end_time_1 = time.time()
        elapsed_time = end_time_1 - start_time_1
        print('Epoch:{:03d} ==> LR: {:7f}, Train Acc: {:.2f}, Train Loss: {:.5f}, Valid Acc: {:.2f}, Valid loss: {:.5f}, Elapsed Time:{:.2f} s'.format(epoch, 
                                                                                                                                                       lr, 
                                                                                                                                                       train_acc*100, 
                                                                                                                                                       train_loss, 
                                                                                                                                                       valid_acc*100, 
                                                                                                                                                       valid_loss, 
                                                                                                                                                       elapsed_time))
    open(path+'history','w').write(str(history))
    print('\nLoss: {}'.format(reports['valid loss'] ))
    
    return reports['valid loss']



from hyperopt import fmin, tpe, Trials, hp
import hyperopt.pyll.stochastic


abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fold_10 = eval(open(abs_path+'/Fold_10.dir').read())
Samples = fold_10['fold-0']['train']+fold_10['fold-0']['valid']
data = make_dataset()
data_list = GetInputData(data.dataframe)

# data split
random.shuffle(Samples)
num_sample = len(Samples)
train_num = int(0.9 * num_sample)
train_samples = Samples[:train_num]
valid_samples = Samples[train_num:]
train_list, valid_list = data_list.split(train_samples=train_samples, valid_samples=valid_samples)


args_dict = {
           'batch_size':hp.choice('batch_size', (32,)),
           'dim':hp.choice('dim', (32,64,128)),
            'mp_step':hp.choice('mp_step', (2,3,4,5)),
           'edge_feat_num':hp.choice('edge_feat_num', (4,)),
           'act_fun':hp.choice('act_fun',  (F.relu, )),
           'lin1_size':hp.choice('lin1_size', (128,512,1024,2048,None)),
           'lin2_size':hp.choice('lin2_size', (128,512,1024,2048,None)),
           'processing_steps':hp.choice('processing_steps', (2,3,4,5)),
           'readout_dense_1':hp.choice('readout_dense_1', (64,128,256,512,1024)),
           'readout_dense_2':hp.choice('readout_dense_2', (64,128,256,512,1024,None)),
           'readout_dense_3':hp.choice('readout_dense_3', (64,128,256,512,1024,None)),
           'readout_act_func':hp.choice('readout_act_func', (F.relu,)),
           'readout_p_dropout':hp.uniform('readout_p_dropout', 0.0, 0.75)
           }

trials = Trials()
best = fmin(
            fn=black_box_function,
            space=args_dict,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            trials_save_file='trials_save_file-mpnn')
print('\nbest:')
print(best)