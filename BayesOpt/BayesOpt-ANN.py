import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow as tf
from ccgnet import experiment as exp
from ccgnet.finetune import *
import random
from ccgnet import layers
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score
from ccgnet.Dataset import Dataset, DataLoader



def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))


def make_dataset():
    data1 = Dataset(abs_path+'/CC_Table/CC_Table.tab', mol_blocks_dir=abs_path+'/Mol_Blocks.dir')
    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
    return data1


def build_model(
                layer_1_size, 
                layer_2_size,
                layer_3_size,
                layer_4_size, 
                layer_5_size,
                layer_6_size,
                layer_7_size,
                act_func, 
                dropout,
               ):
    
    class DNN_5(object):
        def build_model(self, inputs, is_training, global_step=None):
            desc = inputs[0]
            labels = inputs[1]
            tags = inputs[2]
            
            with tf.compat.v1.variable_scope('FC_1') as scope:
                desc = tf.compat.v1.layers.dense(desc, layer_1_size)
                desc = tf.compat.v1.layers.batch_normalization(desc, training=is_training)
                desc = act_func(desc)
                desc = tf.compat.v1.layers.dropout(desc, dropout, training=is_training)
            if layer_2_size != None:
                with tf.compat.v1.variable_scope('FC_2') as scope:
                    desc = tf.compat.v1.layers.dense(desc, layer_2_size)
                    desc = tf.compat.v1.layers.batch_normalization(desc, training=is_training)
                    desc = act_func(desc)
                    desc = tf.compat.v1.layers.dropout(desc, dropout, training=is_training)
            if layer_3_size != None:
                with tf.compat.v1.variable_scope('FC_3') as scope:
                    desc = tf.compat.v1.layers.dense(desc, layer_3_size)
                    desc = tf.compat.v1.layers.batch_normalization(desc, training=is_training)
                    desc = act_func(desc)
                    desc = tf.compat.v1.layers.dropout(desc, dropout, training=is_training)
            if layer_4_size != None:
                with tf.compat.v1.variable_scope('FC_4') as scope:
                    desc = tf.compat.v1.layers.dense(desc, layer_4_size)
                    desc = tf.compat.v1.layers.batch_normalization(desc, training=is_training)
                    desc = act_func(desc)
                    desc = tf.compat.v1.layers.dropout(desc, dropout, training=is_training)
            if layer_5_size != None:
                with tf.compat.v1.variable_scope('FC_5') as scope:
                    desc = tf.compat.v1.layers.dense(desc, layer_5_size)
                    desc = tf.compat.v1.layers.batch_normalization(desc, training=is_training)
                    desc = act_func(desc)
                    desc = tf.compat.v1.layers.dropout(desc, dropout, training=is_training)
            if layer_6_size != None:
                with tf.compat.v1.variable_scope('FC_6') as scope:
                    desc = tf.compat.v1.layers.dense(desc, layer_6_size)
                    desc = tf.compat.v1.layers.batch_normalization(desc, training=is_training)
                    desc = act_func(desc)
                    desc = tf.compat.v1.layers.dropout(desc, dropout, training=is_training)
            if layer_7_size != None:
                with tf.compat.v1.variable_scope('FC_7') as scope:
                    desc = tf.compat.v1.layers.dense(desc, layer_7_size)
                    desc = tf.compat.v1.layers.batch_normalization(desc, training=is_training)
                    desc = act_func(desc)
                    desc = tf.compat.v1.layers.dropout(desc, dropout, training=is_training)
                    
            desc = layers.make_fc_layer(desc, 2, is_training=is_training, with_bn=False, act_func=None, name='final')
            return desc, labels
    return DNN_5()


def black_box_function(args_dict):
    
    tf.reset_default_graph()
    batch_size = args_dict['batch_size']
    layer_1_size = args_dict['layer_1_size']
    layer_2_size = args_dict['layer_2_size']
    layer_3_size = args_dict['layer_3_size']
    layer_4_size = args_dict['layer_4_size']
    layer_5_size = args_dict['layer_5_size']
    layer_6_size = args_dict['layer_6_size']
    layer_7_size = args_dict['layer_7_size']
    act_func = args_dict['act_fun']
    dropout = args_dict['dropout']
    
    # make save dir
    snapshot_path = abs_path+'/bayes_snapshot/'
    model_name = 'BayesOpt-ANN/'
    verify_dir_exists(snapshot_path+model_name)
    if os.listdir(snapshot_path+model_name) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(snapshot_path+model_name) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_)+1)
    # training
    model = build_model(
                        layer_1_size, 
                        layer_2_size,
                        layer_3_size,
                        layer_4_size, 
                        layer_5_size,
                        layer_6_size,
                        layer_7_size,
                        act_func, 
                        dropout,
                       )
    
    model = exp.Model(model, train_data, valid_data, snapshot_path=snapshot_path, 
                      use_subgraph=False, use_desc=False, build_fc=True, model_name=model_name, 
                      dataset_name=dataset_name+'/time_0')
    history = model.fit(num_epoch=100, save_info=True, silence=0, train_batch_size=batch_size,
                        metric='loss')
    loss = min(history['valid_cross_entropy'])
    tf.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    print(args_dict)
    return loss



from hyperopt import fmin, tpe, hp, Trials
import hyperopt.pyll.stochastic


args_dict = {
           'batch_size':hp.choice('batch_size', (64,128,256)),
           'layer_1_size':hp.choice('layer_1_size', (16,32,64,128,256,512)),
           'layer_2_size':hp.choice('layer_2_size', (16,32,64,128,256,512,None)),
           'layer_3_size':hp.choice('layer_3_size', (16,32,64,128,256,512,None)),
           'layer_4_size':hp.choice('layer_4_size', (16,32,64,128,256,512,None)),
           'layer_5_size':hp.choice('layer_5_size', (16,32,64,128,256,512,None)),
           'layer_6_size':hp.choice('layer_6_size', (16,32,64,128,256,512,None)),
           'layer_7_size':hp.choice('layer_7_size', (16,32,64,128,256,512,None)),
           'act_fun':hp.choice('act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
           'dropout':hp.uniform('pred_dropout_rate', 0.0, 0.75)  
           }

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
train_data = [train_data[6], train_data[2], train_data[5]]
valid_data = [valid_data[6], valid_data[2], valid_data[5]]

trials = Trials()
best = fmin(
            fn=black_box_function,
            space=args_dict,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            trials_save_file='trials_save_file-ANN')
print('best:')
print(best)
possible_args_dict = {
                       'batch_size':(64,128,256),
                       'layer_1_size':(16,32,64,128,256,512),
                       'layer_2_size':(16,32,64,128,256,512,None),
                       'layer_3_size':(16,32,64,128,256,512,None),
                       'layer_4_size':(16,32,64,128,256,512,None),
                       'layer_5_size':(16,32,64,128,256,512,None),
                       'layer_6_size':(16,32,64,128,256,512,None),
                       'layer_7_size':(16,32,64,128,256,512,None),
                       'act_fun':(tf.nn.relu, tf.nn.elu, tf.nn.tanh),
                       'dropout':hp.uniform('pred_dropout_rate', 0.0, 0.75)
                       }
print('\n'+str({k:possible_args_dict[k][best[k]] for k in best}))