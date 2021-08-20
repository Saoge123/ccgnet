import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow as tf
import collections
import deepchem as dc
import numpy as np
from rdkit import Chem
from ccdc import io
from deepchem.data import NumpyDataset, pad_features
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.trans import undo_transforms
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization, Lambda
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
import numpy as np
import random



class GetDeepChemCocrystalDataset(object):
    def __init__(self, mol_blocks='Mol_Blocks.dir', cc_table='CC_Table/CC_Table.tab', removeHs=False):
        
        table = [line.strip().split('\t') for line in open(cc_table).readlines()]
        mol_block = eval(open(mol_blocks).read())
        convf = ConvMolFeaturizer()
        exception = {'BOVQUY','CEJPAK','GAWTON','GIPTAA','IDIGUY','LADBIB01','PIGXUY','SIFBIT','SOJZEW','TOFPOW',
                     'QOVZIK','RIJNEF','SIBFAK','SIBFEO','TOKGIJ','TOKGOP','TUQTEE','BEDZUF'}
        self.mol_obj_dic = {}
        for line in table:
            tag = line[-1]
            if tag in exception:
                continue
            c1 = Chem.MolFromMolBlock(mol_block[line[0]], removeHs=removeHs)
            c2 = Chem.MolFromMolBlock(mol_block[line[1]], removeHs=removeHs)
            if c1 == None:
                csd_mol = io.MoleculeReader(mol_block[line[0]])[0].components[0]
                c1 = Chem.MolFromMol2Block(csd_mol.to_string('mol2'), removeHs=removeHs)
            if c2 == None:
                csd_mol = io.MoleculeReader(mol_block[line[1]])[0].components[0]
                c2 = Chem.MolFromMol2Block(csd_mol.to_string('mol2'), removeHs=removeHs)
            y = int(line[2])
            self.mol_obj_dic.setdefault(tag, [convf._featurize(c1), convf._featurize(c2), y])
        
    def dataset_generator(self, sample_list, data_aug=False):
        if data_aug:
            X_, Y_, Tags_ = [], [], []
            X_aug, Y_aug, Tags_aug = [], [], []
            for i in sample_list:
                x = [self.mol_obj_dic[i][0], self.mol_obj_dic[i][1]]
                y = self.mol_obj_dic[i][-1]
                X_.append(x)
                Y_.append(y)
                Tags_.append(i[0])
                x_aug = [self.mol_obj_dic[i][1], self.mol_obj_dic[i][0]]
                y_aug = self.mol_obj_dic[i][-1]
                X_aug.append(x_aug)
                Y_aug.append(y_aug)
                Tags_aug.append(i[0])
            X = X_ + X_aug
            Y = Y_ + Y_aug
            Tags = Tags_ + Tags_aug
            return dc.data.NumpyDataset(X=np.array(X), y=np.array(Y))
        else:
            X, Y, Tags = [], [], []
            for i in sample_list:
                x = [self.mol_obj_dic[i][0], self.mol_obj_dic[i][1]]
                y = self.mol_obj_dic[i][-1]
                X.append(x)
                Y.append(y)
                Tags.append(i)
            
            dataset = dc.data.NumpyDataset(X=np.array(X), y=np.array(Y))
            dataset.tags = Tags
            return dataset


from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix
import time

class TrimGraphOutput(tf.keras.layers.Layer):
    """Trim the output to the correct number of samples.

    GraphGather always outputs fixed size batches.  This layer trims the output
    to the number of samples that were in the actual input tensors.
    """

    def __init__(self, **kwargs):
        super(TrimGraphOutput, self).__init__(**kwargs)

    def call(self, inputs):
        n_samples = tf.squeeze(inputs[1])
        return inputs[0][0:n_samples]

class GraphConvModel(KerasModel):

    def __init__(self,
                   n_tasks,
                   batch_size=100,
                   graph_conv_layers_1=None,
                   graph_conv_layers_2=None,
                   graph_conv_layers_3=None,
                   graph_conv_act_fun=None,
                   graph_gather_act_fun=None,
                   merge='add',
                   dense_layer_1=128,
                   dense_layer_2=128,
                   dense_layer_3=128,
                   dense_act_fun=None,
                   dense_layer_dropout=0.0,
                   mode="classification",
                   number_atom_features=75,
                   n_classes=2,
                   uncertainty=False,
                   **kwargs):
        
        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'")
        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.graph_conv_layers_1 = graph_conv_layers_1
        self.graph_conv_layers_2 = graph_conv_layers_2
        self.graph_conv_layers_3 = graph_conv_layers_3
        self.graph_conv_act_fun = graph_conv_act_fun
        self.graph_gather_act_fun = graph_gather_act_fun
        self.merge = merge
        self.dense_layer_1 = dense_layer_1
        self.dense_layer_2 = dense_layer_2
        self.dense_layer_3 = dense_layer_3
        self.dense_act_fun = dense_act_fun
        self.dense_layer_dropout = dense_layer_dropout
        self.uncertainty = uncertainty
        self.number_atom_features = number_atom_features
        if uncertainty:
            if mode != "regression":
                raise ValueError("Uncertainty is only supported in regression mode")
            if any(d == 0.0 for d in dropout):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty')

        # Build the model.
        atom_features = Input(shape=(self.number_atom_features,))
        degree_slice = Input(shape=(2,), dtype=tf.int32)
        membership = Input(shape=tuple(), dtype=tf.int32)
        n_samples = Input(shape=tuple(), dtype=tf.int32)
        dropout_switch = tf.keras.Input(shape=tuple())

        self.deg_adjs = []
        for i in range(0, 10 + 1):
            deg_adj = Input(shape=(i + 1,), dtype=tf.int32)
            self.deg_adjs.append(deg_adj)
        in_layer = atom_features
        
        gc1_in = [in_layer, degree_slice, membership] + self.deg_adjs
        gc1 = layers.GraphConv(self.graph_conv_layers_1, activation_fn=self.graph_conv_act_fun)(gc1_in)
        batch_norm1 = BatchNormalization(fused=False)(gc1)
        gp_in = [batch_norm1, degree_slice, membership] + self.deg_adjs
        in_layer = layers.GraphPool()(gp_in)
        
        if self.graph_conv_layers_2 != None:
            gc2_in = [in_layer, degree_slice, membership] + self.deg_adjs
            gc2 = layers.GraphConv(self.graph_conv_layers_2, activation_fn=self.graph_conv_act_fun)(gc2_in)
            batch_norm2 = BatchNormalization(fused=False)(gc2)
            gp_in = [batch_norm2, degree_slice, membership] + self.deg_adjs
            in_layer = layers.GraphPool()(gp_in)
        
        if self.graph_conv_layers_3 != None:
            gc3_in = [in_layer, degree_slice, membership] + self.deg_adjs
            gc3 = layers.GraphConv(self.graph_conv_layers_3, activation_fn=self.graph_conv_act_fun)(gc3_in)
            batch_norm3 = BatchNormalization(fused=False)(gc3)
            gp_in = [batch_norm3, degree_slice, membership] + self.deg_adjs
            in_layer = layers.GraphPool()(gp_in)
        
        # Readout
        self.neural_fingerprint = layers.GraphGather(
                                                     batch_size=self.batch_size*2,
                                                     activation_fn=self.graph_gather_act_fun
                                                     )([in_layer, degree_slice, membership] + self.deg_adjs)
        
        readout_dim = self.neural_fingerprint.shape[-1].value
        if self.merge == 'add':
            def reshapes(embed1):
                embed = tf.reshape(embed1, [-1, 2, readout_dim])
                return embed
            readout = Lambda(reshapes)(self.neural_fingerprint)
            
            def resuce_sum(embed1):
                embed = tf.reduce_sum(embed1, 1)
                return embed
            readout = Lambda(resuce_sum)(readout)
        if self.merge == 'concat':
            def reshapes(embed1):
                embed = tf.reshape(embed1, [-1, readout_dim*2])
                return embed
            readout = Lambda(reshapes)(self.neural_fingerprint)
            
        # Foraward
        dense1 = Dense(self.dense_layer_1, activation=self.dense_act_fun)(readout)
        bn1 = BatchNormalization(fused=False)(dense1)
        out = layers.SwitchedDropout(rate=self.dense_layer_dropout)([bn1, dropout_switch])
        
        if self.dense_layer_2 != None:
            dense2 = Dense(self.dense_layer_2, activation=self.dense_act_fun)(out)
            bn2 = BatchNormalization(fused=False)(dense2)
            out = layers.SwitchedDropout(rate=self.dense_layer_dropout)([bn2, dropout_switch])
            
        if self.dense_layer_3 != None:
            dense3 = Dense(self.dense_layer_3, activation=self.dense_act_fun)(out)
            bn3 = BatchNormalization(fused=False)(dense3)
            out = layers.SwitchedDropout(rate=self.dense_layer_dropout)([bn3, dropout_switch])
        
        n_tasks = self.n_tasks
        if self.mode == 'classification':
            n_classes = self.n_classes
            logits = Reshape((n_tasks, n_classes))(Dense(n_tasks * n_classes)(out))
            logits = TrimGraphOutput()([logits, n_samples])
            output = Softmax()(logits)
            outputs = [output, logits]
            output_types = ['prediction', 'loss']
            loss = SoftmaxCrossEntropy()
        else:
            output = Dense(n_tasks)(self.neural_fingerprint)
            output = TrimGraphOutput()([output, n_samples])
            if self.uncertainty:
                log_var = Dense(n_tasks)(out)
                log_var = TrimGraphOutput()([log_var, n_samples])
                var = Activation(tf.exp)(log_var)
                outputs = [output, var, output, log_var]
                output_types = ['prediction', 'variance', 'loss', 'loss']

                def loss(outputs, labels, weights):
                    diff = labels[0] - outputs[0]
                    return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
            else:
                outputs = [output]
                output_types = ['prediction']
                loss = L2Loss()
        model = tf.keras.Model(
            inputs=[
                     atom_features, degree_slice, membership, n_samples, dropout_switch
                   ] + self.deg_adjs,
                    outputs=outputs)
        super(GraphConvModel, self).__init__(
            model, loss, output_types=output_types, batch_size=self.batch_size, **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches):

                X_b = X_b.reshape(-1)
                if self.mode == 'classification':
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                           -1, self.n_tasks, self.n_classes)
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                n_samples = np.array(y_b.shape[0])   #####
                if mode == 'predict':
                    dropout = np.array(0.0)
                else:
                    dropout = np.array(1.0)
                inputs = [
                    multiConvMol.get_atom_features(), multiConvMol.deg_slice,
                    np.array(multiConvMol.membership), n_samples, dropout
                         ]
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
                yield (inputs, [y_b], [w_b])
    
    def fit_epoch(self, 
            train_set,
            valid_set,
            test_set=None,
            ecc_test_set=None,
            epochs=1,
            metric=accuracy_score,
            save_metric='acc',  # 'loss'
            mode='fit',
            silence=False,
            deterministic=False,
            pad_batches=True,
            eps=1e-15,
            with_early_stop=False,
            early_stop_cutoff=20,
            model_dir='./snapshot/BayesOpt-GCN/'):
        
        
        best_valid = float('inf') if save_metric=='loss' else 0.0
        self.Best_Val_Pred = None
        count = 0
        for epoch in range(epochs):
            beg_time = time.time()
            train_loss = self.fit(train_set, nb_epoch=1, deterministic=deterministic)
            train_logits = self.predict(train_set)
            train_logits = train_logits.reshape(-1,2)
            train_pred = np.argmax(train_logits, axis=1)
            train_acc = metric(train_set.y, train_pred)
            
            ##### Valid #####
            valid_logits = self.predict(valid_set).reshape(-1,2)
            valid_logits = np.clip(valid_logits, eps, 1 - eps)
            valid_loss = np.average(-np.sum(np.multiply(to_one_hot(valid_set.y),np.log(valid_logits)),1))
            valid_pred = np.argmax(valid_logits, axis=1)
            valid_acc = metric(valid_set.y, valid_pred)
            
            if save_metric == 'acc':
                is_save = valid_acc > best_valid
            elif save_metric == 'loss':
                is_save = valid_loss < best_valid
            if is_save:
                count = 0
                self.save_checkpoint(max_checkpoints_to_keep=2, model_dir=model_dir)
                best_valid = valid_loss if save_metric=='loss' else valid_acc
                self.Best_Val_Pred = valid_logits
                valid_info = [
                    'step:{}'.format(epoch),
                    'valid_acc:{}'.format(valid_acc),
                    'valid_cross_entropy:{}'.format(valid_loss),
                    'train_acc:{}'.format(train_acc),
                    'train_cross_entropy:{}'.format(train_loss),
                    str(valid_set.y.tolist()), 
                    str(valid_logits.tolist()), 
                    str(valid_set.tags)
                    ]
                open(model_dir+'/model-{}_info.txt'.format(epoch), 'w').writelines('\n'.join(valid_info))
                if test_set is not None:
                    test_logits = self.predict(test_set).reshape(-1,2)
                    self.Test_Pred = test_logits
                    test_pred = np.argmax(test_logits, axis=1)
                    test_acc = metric(test_set.y, test_pred)
                    test_info = [
                        str(test_acc),
                        str(test_set.y.tolist()),
                        str(test_logits.tolist()),
                        str(test_set.tags)
                        ]
                    open(model_dir+'/model-val-info.txt', 'w').writelines('\n'.join(test_info))
                if ecc_test_set is not None:
                    ecc_test_logits = self.predict(ecc_test_set).reshape(-1,2)
                    ecc_test_pred = np.argmax(ecc_test_logits, axis=1)
                    ecc_test_acc = metric(ecc_test_set.y, ecc_test_pred)
                    ecc_test_info = [
                        str(ecc_test_acc),
                        str(ecc_test_set.y.tolist()),
                        str(ecc_test_logits.tolist()),
                        str(ecc_test_set.tags)
                        ]
                    open(model_dir+'/model-eccval-info.txt', 'w').writelines('\n'.join(ecc_test_info))
            else:
                count += 1
            end_time = time.time()
            time_gap = end_time - beg_time
            if silence == False:
                best_valid_ = round(best_valid*100, 3) if save_metric == 'acc' else round(best_valid, 5)
                print_contents = 'Epoch {:03d} ==> Train_Acc:{:.3f}; Train_Loss:{:.5f} Valid_Acc:{:.3f}; Valid_Loss:{}; Best_Valid:{}; Elapsed_Time:{:.2f}'
                print_contents = print_contents.format(epoch, train_acc*100, train_loss, valid_acc*100, round(valid_loss,5), best_valid_, time_gap)
                print(print_contents)
            if with_early_stop:
                if count == early_stop_cutoff:
                    print('Early Stopping ......')
                    break
        return best_valid

           
def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def black_box_function(args_dict):
    
    tf.reset_default_graph()
    graph_conv_layers_1 = args_dict['graph_conv_layers_1']
    graph_conv_layers_2 = args_dict['graph_conv_layers_2']
    graph_conv_layers_3 = args_dict['graph_conv_layers_3']
    graph_conv_act_fun = args_dict['graph_conv_act_fun']
    graph_gather_act_fun = args_dict['graph_gather_act_fun']
    merge = args_dict['merge']
    dense_layer_1 = args_dict['dense_layer_1']
    dense_layer_2 = args_dict['dense_layer_2']
    dense_layer_3 = args_dict['dense_layer_3']
    dense_act_fun = args_dict['dense_act_fun']
    dense_layer_dropout = args_dict['dense_layer_dropout']
    # make save dir
    snapshot_path = abs_path+'/bayes_snapshot/'
    model_name = '/BayesOpt-GCN/'
    verify_dir_exists(snapshot_path+model_name)
    if os.listdir(snapshot_path+model_name) == []:
            dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(snapshot_path+model_name) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_)+1)
    path_to_save = snapshot_path + model_name + dataset_name + '/time_0'
    verify_dir_exists(path_to_save)
    # build model
    m = GraphConvModel(
                       1, 
                       batch_size=128,
                       graph_conv_layers_1=graph_conv_layers_1,
                       graph_conv_layers_2=graph_conv_layers_2,
                       graph_conv_layers_3=graph_conv_layers_3,
                       graph_conv_act_fun=graph_conv_act_fun,
                       graph_gather_act_fun=graph_gather_act_fun,
                       merge=merge,
                       dense_layer_1=dense_layer_1,
                       dense_layer_2=dense_layer_2,
                       dense_layer_3=dense_layer_3,
                       dense_act_fun=dense_act_fun,
                       dense_layer_dropout=dense_layer_dropout
                       )
    # training
    loss = m.fit_epoch(train_set, 
                       valid_set, 
                       test_set=None, 
                       ecc_test_set=None, 
                       epochs=100, 
                       deterministic=False, 
                       save_metric='loss',
                       model_dir=path_to_save)   #, checkpoint_interval=100)
    tf.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    print(str(args_dict))
    return loss




from hyperopt import fmin, tpe, hp, Trials
import hyperopt.pyll.stochastic

abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fold_10 = eval(open(abs_path+'/Fold_10.dir').read())
CCDataset = GetDeepChemCocrystalDataset(mol_blocks=abs_path+'/Mol_Blocks.dir', cc_table=abs_path+'/CC_Table/CC_Table.tab')
Samples = fold_10['fold-0']['train']+fold_10['fold-0']['valid']

# data spliting
random.shuffle(Samples)
num_sample = len(Samples)
train_num = int(0.9 * num_sample)
train_samples = Samples[:train_num]
valid_samples = Samples[train_num:]
train_set = CCDataset.dataset_generator(train_samples)
valid_set = CCDataset.dataset_generator(valid_samples)

args_dict = {
            'graph_conv_layers_1':hp.choice('graph_conv_layers_1', (64,128,256)),
            'graph_conv_layers_2':hp.choice('graph_conv_layers_2', (64,128,256,None)),
            'graph_conv_layers_3':hp.choice('graph_conv_layers_3', (64,128,256,None)),
            'graph_conv_act_fun':hp.choice('graph_conv_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
            'graph_gather_act_fun':hp.choice('graph_gather_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
            'merge':hp.choice('merge',('add', 'concat')),
            'dense_layer_1':hp.choice('dense_layer_1', (64,128,256,512,1024)),
            'dense_layer_2':hp.choice('dense_layer_2', (64,128,256,512,1024,None)),
            'dense_layer_3':hp.choice('dense_layer_3', (64,128,256,512,1024,None)),
            'dense_act_fun':hp.choice('dense_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
            'dense_layer_dropout':hp.uniform('dense_layer_dropout', 0.0, 0.75),
            }

trials = Trials()
best = fmin(
            fn=black_box_function,
            space=args_dict,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            trials_save_file='trials_save_file-GCN')
print('best:')
print(best)