import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow as tf
from ccgnet import experiment as exp
from ccgnet.finetune import *
from ccgnet import layers
from ccgnet.layers import *
import numpy as np
import time
import random
from sklearn.metrics import balanced_accuracy_score
from ccgnet.Dataset import Dataset, DataLoader
from Featurize.Coformer import Coformer
from Featurize.Cocrystal import Cocrystal



def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))


def make_dataset():
    data1 = Dataset(abs_path+'/CC_Table/CC_Table.tab', mol_blocks_dir=abs_path+'/Mol_Blocks.dir')
    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
    return data1


def build_model(
                blcok_1_size, 
                blcok_2_size,
                blcok_3_size,
                blcok_4_size, 
                blcok_5_size, 
                mp_act_func,
                n_head, 
                pred_layer_1_size, 
                pred_layer_2_size,
                pred_layer_3_size,
                pred_act_func, 
                pred_dropout_rate
               ):
    
    class Model(object):
        def build_model(self, inputs, is_training, global_step=None):
            V = inputs[0]
            A = inputs[1]
            labels = inputs[2]
            mask = inputs[3]
            graph_size = inputs[4]
            tags = inputs[5]
            global_state = inputs[6]
            subgraph_size = inputs[7]
            
            V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_1_size, act_func=mp_act_func, mask=mask, num_updates=global_step, is_training=is_training)
            if blcok_2_size != None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_2_size, act_func=mp_act_func, mask=mask, num_updates=global_step, is_training=is_training)
            if blcok_3_size != None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_3_size, act_func=mp_act_func, mask=mask, num_updates=global_step, is_training=is_training)
            if blcok_4_size != None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_4_size, act_func=mp_act_func, mask=mask, num_updates=global_step, is_training=is_training)
            if blcok_5_size != None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_5_size, act_func=mp_act_func, mask=mask, num_updates=global_step, is_training=is_training)
            # Readout
            V = ReadoutFunction(V, global_state, graph_size, num_head=n_head, is_training=is_training)
            # Prediction
            with tf.compat.v1.variable_scope('Predictive_FC_1') as scope:
                V = layers.make_embedding_layer(V, pred_layer_1_size)
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = pred_act_func(V)
                V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)
            if pred_layer_2_size != None:
                with tf.compat.v1.variable_scope('Predictive_FC_2') as scope:
                    V = layers.make_embedding_layer(V, pred_layer_2_size)
                    V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                    V = pred_act_func(V)
                    V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)
            if pred_layer_3_size != None:
                with tf.compat.v1.variable_scope('Predictive_FC_3') as scope:
                    V = layers.make_embedding_layer(V, pred_layer_3_size)
                    V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                    V = pred_act_func(V)
                    V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)
            # Output
            out = layers.make_embedding_layer(V, 2, name='final')
            return out, labels
    return Model()

def black_box_function(args_dict):
    print('\n'+str(args_dict))
    tf.reset_default_graph()
    batch_size = args_dict['batch_size']
    blcok_1_size = args_dict['blcok_1_size']
    blcok_2_size = args_dict['blcok_2_size']
    blcok_3_size = args_dict['blcok_3_size']
    blcok_4_size = args_dict['blcok_4_size'] 
    blcok_5_size = args_dict['blcok_5_size']
    mp_act_func = args_dict['mp_act_func']
    n_head = args_dict['n_head']
    pred_layer_1_size = args_dict['pred_layer_1_size']
    pred_layer_2_size = args_dict['pred_layer_2_size']
    pred_layer_3_size = args_dict['pred_layer_3_size']
    pred_act_func = args_dict['pred_act_func']
    pred_dropout_rate = args_dict['pred_dropout_rate']
    
    # make save dir
    snapshot_path = abs_path+'/bayes_snapshot/'
    model_name = 'BayesOpt-CCGNet/'
    verify_dir_exists(snapshot_path+model_name)
    if os.listdir(snapshot_path+model_name) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(snapshot_path+model_name) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_)+1)
    # training
    model = build_model(
                        blcok_1_size, 
                        blcok_2_size,
                        blcok_3_size,
                        blcok_4_size, 
                        blcok_5_size, 
                        mp_act_func,
                        n_head, 
                        pred_layer_1_size, 
                        pred_layer_2_size,
                        pred_layer_3_size,
                        pred_act_func, 
                        pred_dropout_rate
                       )
    
    model = exp.Model(model, train_data, valid_data, with_test=False, snapshot_path=snapshot_path, use_subgraph=True,
                      model_name=model_name, dataset_name=dataset_name+'/time_0')
    history = model.fit(num_epoch=100, save_info=True, save_att=False, silence=False, train_batch_size=batch_size, 
                        max_to_keep=1, metric='loss')
    loss = min(history['valid_cross_entropy'])
    tf.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    return loss




from hyperopt import fmin, tpe, Trials, hp
import hyperopt.pyll.stochastic


abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fold_10 = eval(open(abs_path+'/Fold_10.dir').read())
data = make_dataset()
Samples = fold_10['fold-0']['train']+fold_10['fold-0']['valid']

# data spliting
random.shuffle(Samples)
num_sample = len(Samples)
train_num = int(0.9 * num_sample)
train_samples = Samples[:train_num]
valid_samples = Samples[train_num:]
train_data, valid_data = data.split(train_samples=train_samples, valid_samples=valid_samples)

args_dict = {
           'batch_size':hp.choice('batch_size', (128,)),
           'blcok_1_size':hp.choice('blcok_1_size', (16,32,64,128,256)),
           'blcok_2_size':hp.choice('blcok_2_size', (16,32,64,128,256,None)),
           'blcok_3_size':hp.choice('blcok_3_size', (16,32,64,128,256,None)),
           'blcok_4_size':hp.choice('blcok_4_size', (16,32,64,128,256,None)),
           'blcok_5_size':hp.choice('blcok_5_size', (16,32,64,128,256,None)),
           'mp_act_func':hp.choice('mp_act_func', (tf.nn.relu, )),
           'n_head':hp.choice('n_head', (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
           'pred_layer_1_size':hp.choice('pred_layer_1_size', (64,128,256,512,1024)),
           'pred_layer_2_size':hp.choice('pred_layer_2_size', (64,128,256,512,1024,None)),
           'pred_layer_3_size':hp.choice('pred_layer_3_size', (64,128,256,512,1024,None)),
           'pred_act_func':hp.choice('pred_act_func', (tf.nn.relu, )),
           'pred_dropout_rate':hp.uniform('pred_dropout_rate', 0.0, 0.75)
           }

trials = Trials()
best = fmin(
            fn=black_box_function,
            space=args_dict,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            trials_save_file='trials_save_file-ccgnet')
print('\nbest:')
print(best)