import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ccgnet import experiment as exp
from ccgnet import layers
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score
from ccgnet.Dataset import Dataset, DataLoader



def build_model(
                graphcnn_layer_1_size, 
                graphcnn_layer_2_size,
                graphcnn_layer_3_size,
                graphcnn_act_fun,
                graph_pool_1_size,
                graph_pool_2_size,
                graph_pool_3_size,
                graph_pool_act_fun,
                dense_layer_1_size, 
                dense_layer_2_size,
                dense_layer_3_size,
                dense_act_func, 
                dense_dropout
               ):
    mask_judge = (graph_pool_1_size, graph_pool_2_size, graph_pool_3_size)
    print(mask_judge)
    class Model(object):
        def build_model(self, inputs, is_training, global_step):
            V = inputs[0]
            A = inputs[1]
            labels = inputs[2]
            mask = inputs[3]
            graph_size = inputs[4]
            tags = inputs[5]
            # Graph-CNN stage
            V = layers.make_graphcnn_layer(V, A, graphcnn_layer_1_size)
            V = layers.make_bn(V, is_training, mask=mask, num_updates=global_step)
            V = graphcnn_act_fun(V)
            if graph_pool_1_size != None:
                V_pool, A = layers.make_graph_embed_pooling(V, A, mask=mask, no_vertices=graph_pool_1_size)
                V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
                V = graph_pool_act_fun(V)
            if graphcnn_layer_2_size != None:
                if mask_judge[0] != None:
                    m = None
                else:
                    m = mask
                V = layers.make_graphcnn_layer(V, A, graphcnn_layer_2_size)
                V = layers.make_bn(V, is_training, mask=m, num_updates=global_step)
                V = graphcnn_act_fun(V)
            if graph_pool_2_size != None:
                if mask_judge[0] != None:
                    m = None
                else:
                    m = mask
                V_pool, A = layers.make_graph_embed_pooling(V, A, mask=m, no_vertices=graph_pool_2_size)
                V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
                V = graph_pool_act_fun(V)
            if graphcnn_layer_3_size != None:
                if mask_judge[1] != None or mask_judge[0] != None:
                    m = None
                else:
                    m = mask
                V = layers.make_graphcnn_layer(V, A, graphcnn_layer_3_size)
                V = layers.make_bn(V, is_training, mask=m, num_updates=global_step)
                V = graphcnn_act_fun(V)
                
            if mask_judge[1] != None or mask_judge[0] != None:
                m = None
            else:
                m = mask
            V_pool, A = layers.make_graph_embed_pooling(V, A, mask=m, no_vertices=graph_pool_3_size)
            V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
            V = graph_pool_act_fun(V)
            
            # Predictive Stage
            no_input_features = int(np.prod(V.get_shape()[1:]))
            V = tf.reshape(V, [-1, no_input_features])
            V = layers.make_embedding_layer(V, dense_layer_1_size, name='FC-1')
            V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
            V = dense_act_func(V)
            V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)
            if dense_layer_2_size != None:
                V = layers.make_embedding_layer(V, dense_layer_2_size, name='FC-2')
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = dense_act_func(V)
                V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)
            if dense_layer_3_size != None:
                V = layers.make_embedding_layer(V, dense_layer_3_size, name='FC-3')
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = dense_act_func(V)
                V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)
            
            out = layers.make_embedding_layer(V, 2, name='final')
            return out, labels
    return Model()

def black_box_function(args_dict):
    
    tf.reset_default_graph()
    batch_size = args_dict['batch_size']
    graphcnn_layer_1_size = args_dict['graphcnn_layer_1_size']
    graphcnn_layer_2_size = args_dict['graphcnn_layer_2_size']
    graphcnn_layer_3_size = args_dict['graphcnn_layer_3_size']
    graphcnn_act_fun = args_dict['graphcnn_act_fun']
    graph_pool_1_size = args_dict['graph_pool_1_size']
    graph_pool_2_size = args_dict['graph_pool_2_size']
    graph_pool_3_size = args_dict['graph_pool_3_size']
    graph_pool_act_fun = args_dict['graph_pool_act_fun']
    dense_layer_1_size = args_dict['dense_layer_1_size']
    dense_layer_2_size = args_dict['dense_layer_2_size']
    dense_layer_3_size = args_dict['dense_layer_3_size']
    dense_act_func = args_dict['dense_act_func']
    dense_dropout = args_dict['dense_dropout']
    
    # make save dir
    snapshot_path = abs_path+'/bayes_snapshot/'
    model_name = 'BayesOpt-GraphCNN/'
    verify_dir_exists(snapshot_path+model_name)
    if os.listdir(snapshot_path+model_name) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(snapshot_path+model_name) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_)+1)
    
    model = build_model(graphcnn_layer_1_size, 
                        graphcnn_layer_2_size, 
                        graphcnn_layer_3_size, 
                        graphcnn_act_fun, 
                        graph_pool_1_size, 
                        graph_pool_2_size, 
                        graph_pool_3_size, 
                        graph_pool_act_fun,
                        dense_layer_1_size, 
                        dense_layer_2_size, 
                        dense_layer_3_size, 
                        dense_act_func, 
                        dense_dropout)
    model = exp.Model(model, train_data, valid_data, with_test=False, snapshot_path=snapshot_path, use_subgraph=False, use_desc=False, build_fc=False,
                          model_name=model_name, dataset_name=dataset_name+'/time_0')
    history = model.fit(num_epoch=100, save_info=True, save_att=False, silence=False, train_batch_size=batch_size, 
                        max_to_keep=1, metric='loss')
    loss = min(history['valid_cross_entropy'])
    tf.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    print(str(args_dict))
    return loss


from hyperopt import fmin, tpe, Trials, hp
import hyperopt.pyll.stochastic
import random


def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def make_dataset():
    data1 = Dataset(abs_path+'/CC_Table/CC_Table.tab', mol_blocks_dir=abs_path+'/Mol_Blocks.dir')
    data1.make_graph_dataset(Desc=0, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
    return data1

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
             'graphcnn_layer_1_size':hp.choice('graphcnn_layer_1_size', (16,32,64,128,256)),
             'graphcnn_layer_2_size':hp.choice('graphcnn_layer_2_size', (16,32,64,128,256,None)),
             'graphcnn_layer_3_size':hp.choice('graphcnn_layer_3_size', (16,32,64,128,256,None)),
             'graphcnn_act_fun':hp.choice('graphcnn_act_fun', (tf.nn.relu, )), 
             'graph_pool_1_size':hp.choice('graph_pool_1_size', (8,16,32,None)),
             'graph_pool_2_size':hp.choice('graph_pool_2_size', (8,16,32,None)),
             'graph_pool_3_size':hp.choice('graph_pool_3_size', (8,16,32)),
             'graph_pool_act_fun':hp.choice('graph_pool_act_fun', (tf.nn.relu, )),
             'dense_layer_1_size':hp.choice('dense_layer_1_size', (64,128,256,512)),
             'dense_layer_2_size':hp.choice('dense_layer_2_size', (64,128,256,512,None)),
             'dense_layer_3_size':hp.choice('dense_layer_3_size', (64,128,256,512,None)),
             'dense_act_func':hp.choice('dense_act_func', (tf.nn.relu, )),
             'dense_dropout':hp.uniform('dense_dropout', 0.0, 0.75)
             }
trials = Trials()
best = fmin(
            fn=black_box_function,
            space=args_dict,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            trials_save_file='trials_save_file-graphcnn')
print('\nbest:')
print(best)