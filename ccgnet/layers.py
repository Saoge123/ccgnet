# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math


class GraphCNNKeys(object):
    TRAIN_SUMMARIES = "train_summaries"
    TEST_SUMMARIES = "test_summaries"
    
class GraphCNNGlobal(object):
    BN_DECAY = 0.999
    GRAPHCNN_INIT_FACTOR = 1.
    GRAPHCNN_I_FACTOR = 1.0

def make_variable(name, shape, initializer=tf.truncated_normal_initializer(), regularizer=None):
    dtype = tf.float32
    var = tf.compat.v1.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=dtype)
    return var

def make_bias_variable(name, shape):
    dtype = tf.float32
    var = tf.compat.v1.get_variable(name, shape, initializer=tf.constant_initializer(0.1), dtype=dtype)
    return var

def make_variable_with_weight_decay(name, shape, stddev=0.01, wd=0.0005):
    dtype = tf.float32
    regularizer = None
    if wd is not None and wd > 1e-7:
        def regularizer(var):
            return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    var = make_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), regularizer=regularizer)
    return var

def make_bn(input, phase, axis=-1, epsilon=0.001, mask=None, num_updates=None, name=None):
    default_decay = GraphCNNGlobal.BN_DECAY
    with tf.compat.v1.variable_scope(name, default_name='BatchNorm') as scope:
        input_size = input.get_shape()[axis].value
        if axis == -1:
            axis = len(input.get_shape())-1
        axis_arr = [i for i in range(len(input.get_shape())) if i != axis]
        if mask == None:
            batch_mean, batch_var = tf.nn.moments(input, axis_arr)
        else:
            batch_mean, batch_var = tf.nn.weighted_moments(input, axis_arr, mask)
        gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
        beta = make_bias_variable('bias', input_size)
        ema = tf.train.ExponentialMovingAverage(decay=default_decay, num_updates=num_updates)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)

def batch_mat_mult(A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    if B.get_shape()[1] == None:
        axis_2 = -1
    else:
        axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
    return result
    
def make_softmax_layer(V, axis=1, name=None):
    with tf.compat.v1.variable_scope(name, default_name='Softmax') as scope:
        max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
        exp = tf.exp(tf.subtract(V, max_value))
        prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True), name='prob')
        return prob

def make_graphcnn_layer(V, A, no_filters, no_features_for_conv=None, name=None):
    with tf.compat.v1.variable_scope(name, default_name='Graph-CNN') as scope:
        no_A = A.get_shape()[2].value
        if no_features_for_conv == None:
            no_features = V.get_shape()[2].value
            if no_features == None:
                no_features = no_filters*2
        else:
            no_features = no_features_for_conv
        W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])

        A_shape = tf.shape(A)
        A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
        n = tf.matmul(A_reshape, V)
        n = tf.reshape(n, [-1, A_shape[1], no_A*no_features])
        result = batch_mat_mult(n, W) + batch_mat_mult(V, W_I) + b
        return result

def make_graph_embed_pooling(V, A, no_vertices=1, mask=None, name=None):
    with tf.compat.v1.variable_scope(name, default_name='GraphEmbedPooling') as scope:
        factors = make_embedding_layer(V, no_vertices, name='Factors')
        
        if mask is not None:
            factors = tf.multiply(factors, mask)  
        factors = make_softmax_layer(factors)
        
        result = tf.matmul(factors, V, transpose_a=True)
        
        if no_vertices == 1:
            no_features = V.get_shape()[2].value
            return tf.reshape(result, [-1, no_features]), A
        
        result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
        result_A = tf.matmul(factors, result_A, transpose_a=True)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], no_vertices, A.get_shape()[2].value, no_vertices))
        
        return result, result_A
    
def make_embedding_layer(V, no_filters, name=None):
    with tf.compat.v1.variable_scope(name, default_name='Embed') as scope:
        no_features = V.get_shape()[-1].value
        W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=1.0/math.sqrt(no_features))
        b = make_bias_variable('bias', [no_filters])
        V_reshape = tf.reshape(V, (-1, no_features))
        s = tf.slice(tf.shape(V), [0], [len(V.get_shape())-1])
        s = tf.concat([s, tf.stack([no_filters])], 0)
        result = tf.reshape(tf.matmul(V_reshape, W) + b, s, name='result')
        return result


def make_fc_layer(inputs, out_size, with_act_func=True, with_bn=True, is_training=1, num_updates=None, name=None):
    with tf.compat.v1.variable_scope(name, default_name='FullConnect') as scope:
        input_size = inputs.get_shape()[-1].value
        W = make_variable_with_weight_decay('weights', [input_size,out_size], stddev=1.0/math.sqrt(input_size))
        b = make_bias_variable('bias', [out_size])
        result = tf.matmul(inputs,W) + b
        if with_bn:
            result = make_bn(result, is_training)
        if with_act_func:
            return tf.nn.relu(result)
        else:
            return result

def batch_node_number(numbers):
    with tf.compat.v1.variable_scope('BatchNodeNumber') as scope:
        init_counter = tf.constant(0,dtype=tf.as_dtype(numbers.dtype))
        init_tensor =  tf.zeros(0, dtype=tf.as_dtype(numbers.dtype))
        c = lambda i, m: tf.less(i, tf.size(numbers,out_type=tf.as_dtype(numbers.dtype)))
        b = lambda i, m: [i+1, tf.concat([m,tf.add(tf.zeros(tf.gather_nd(numbers,[i]),tf.as_dtype(numbers.dtype)),i)], axis=0)]
        loop = tf.while_loop(c, b, loop_vars=[init_counter, init_tensor], shape_invariants=[init_counter.get_shape(), tf.TensorShape([None,])])
    return loop[1]

def batch_node_range(numbers):
    with tf.compat.v1.variable_scope('BatchNodeRange') as scope:
        init_counter = tf.constant(0,dtype=tf.as_dtype(numbers.dtype))
        init_tensor =  tf.zeros(0, dtype=tf.as_dtype(numbers.dtype))
        c = lambda i, m: tf.less(i, tf.size(numbers,out_type=tf.as_dtype(numbers.dtype)))
        b = lambda i, m: [i+1, tf.concat([m,tf.range(tf.gather_nd(numbers,[i]))], axis=0)]
        loop = tf.while_loop(c, b, loop_vars=[init_counter, init_tensor], shape_invariants=[init_counter.get_shape(), tf.TensorShape([None,])])
    return loop[1]

def segment_softmax(batch_node_num, gate, name=None):
    with tf.compat.v1.variable_scope(name, default_name='Segment_Softmax') as scope:
        segment_max = tf.math.segment_max(gate, batch_node_num, name='segment_max')
        segment_max_gathered = tf.gather(segment_max, batch_node_num, name='segment_max_gathered')
        subtract = tf.subtract(gate, segment_max_gathered, name='subtract')
        exp = tf.exp(subtract, name='exp')
        segment_sum = tf.add(tf.gather(tf.math.segment_sum(exp, batch_node_num), batch_node_num),1e-16, name='segment_sum')
        attention = tf.divide(exp, segment_sum, name='attention')
        return attention

def broadcast_global_state(subgraph_size, global_state, V_shape, name=None):
    with tf.compat.v1.variable_scope(name, default_name='Broadcast_Global_State') as scope:
        init_counter = tf.constant(0, dtype=tf.as_dtype(subgraph_size.dtype))
        no_global_state_feature = global_state.get_shape()[-1].value
        init_tensor =  tf.zeros((1,V_shape[1],no_global_state_feature), dtype=tf.as_dtype(global_state.dtype))
        max_graph_size = V_shape[1]
        def cond(t, i):
            return tf.less(i, tf.shape(global_state)[0])
        
        def body(t, i):
            subg1 = tf.zeros(tf.gather_nd(subgraph_size[i], [0]), dtype=tf.int32)
            subg2 = tf.ones(tf.gather_nd(subgraph_size[i], [1]), dtype=tf.int32)
            cated = tf.concat([subg1,subg2], axis=0)
            gathered_gs = tf.gather(global_state[i], cated)
            padding_num = tf.subtract(max_graph_size, tf.reduce_sum(subgraph_size[i]))
            padding = tf.pad(gathered_gs, [[0,padding_num],[0,0]])
            padding_reshape = tf.reshape(padding, [1, max_graph_size, no_global_state_feature])
            return tf.concat([t, padding_reshape], axis=0), i+1
        
        loop = tf.while_loop(cond, body, loop_vars=[init_tensor, init_counter], shape_invariants=[tf.TensorShape([None,None,None]), init_counter.get_shape()])
        return loop[0][1:]


def CCGBlock(V, A, global_state, subgraph_size, no_filters=64, act_fun=tf.nn.relu, mask=None, num_updates=None, is_training=1, name=None):
    with tf.compat.v1.variable_scope(name, default_name='CCGBlock') as scope:
        ######## transfer global_state #########
        global_state_transfer = make_fc_layer(global_state, no_filters, with_act_func=True, with_bn=False, is_training=is_training)
        global_state_transfer = tf.reshape(global_state_transfer, [-1, 2, no_filters])
        ######## concat global_state and V #########
        if V.get_shape()[1].value == None or V.get_shape()[2].value ==None:
            V_shape = [None, A.get_shape()[1].value, V.get_shape()[2].value]
        else:
            V_shape = [None, V.get_shape()[1].value, V.get_shape()[2].value]
        bgs = broadcast_global_state(subgraph_size, global_state_transfer, V_shape)
        V = tf.concat([V, bgs], axis=-1)
        ######## Graph Convolution #########
        no_features_for_conv = no_filters + V_shape[-1]
        V = make_graphcnn_layer(V, A, no_filters, no_features_for_conv=no_features_for_conv)
        V = make_bn(V, is_training, mask=mask, num_updates=num_updates)
        V = act_fun(V)
        global_state_transfer = tf.reshape(global_state_transfer, [-1, no_filters])
        global_state_transfer = make_bn(global_state_transfer, is_training, num_updates=num_updates)
        global_state_transfer = act_fun(global_state_transfer)
        return V, global_state_transfer

    
def multi_head_global_attention(V, graph_size, num_head=5, is_training=1, concat=True, multi_layer=None, name=None):
    with tf.compat.v1.variable_scope(name, default_name='Multi_Head_Global_Attention') as scope:
        node_numbers = batch_node_number(graph_size)
        node_range = batch_node_range(graph_size)
        node_index = tf.stack([node_numbers, node_range], axis=1)
        gathered_V = tf.gather_nd(V, node_index)
        input_size = gathered_V.get_shape()[-1].value
        if multi_layer:
            for i in multi_layer:
                gathered_V = make_fc_layer(gathered_V, i, is_training=is_training, with_bn=False)
        weight = make_variable_with_weight_decay('Att_Transform_Weights', [gathered_V.get_shape()[-1].value, num_head*input_size], stddev=1.0/math.sqrt(input_size))
        b = make_bias_variable('Att_Transform_Bias', [num_head*input_size])
        tune_weight = make_variable_with_weight_decay('Att_Tune_Weights', [1,num_head,input_size], stddev=1.0/math.sqrt(input_size))
        gathered_V = tf.matmul(gathered_V, weight) + b
        gathered_V = tf.reshape(gathered_V, (-1, num_head, input_size))
        alpha = tf.multiply(tune_weight, gathered_V)
        alpha = tf.reduce_sum(alpha, axis=-1)
        alpha = tf.nn.leaky_relu(alpha, alpha=0.2)
        alpha = segment_softmax(node_numbers, alpha)
        if concat:
            alpha_collect = tf.reduce_mean(alpha, -1, name='Attentions')
            alpha = tf.reshape(alpha, [-1, num_head, 1])
            V = tf.multiply(gathered_V, alpha)
            V = tf.math.segment_sum(V, node_numbers)
            V =  tf.reshape(V,[-1, num_head*input_size])
            return V
        else:
            alpha_collect = tf.reduce_mean(alpha, -1, name='Attentions')
            alpha = tf.reshape(alpha, [-1, num_head, 1])
            V = tf.multiply(gathered_V, alpha)
            V = tf.reduce_mean(V, 1)
            V = tf.math.segment_sum(V, node_numbers)
            return V


def Set2Set(x, graph_size, time_steps=3, name=None):
    with tf.compat.v1.variable_scope(name, default_name='Set2Set') as scope:
        node_numbers = batch_node_number(graph_size)
        node_range = batch_node_range(graph_size)
        node_index = tf.stack([node_numbers, node_range], axis=1)
        x = tf.gather_nd(x, node_index)
        in_size = x.get_shape()[-1]
        out_size = in_size * 2
        batch_size = tf.reduce_max(node_numbers) + 1
        lstm = tf.nn.rnn_cell.LSTMCell(num_units=in_size)
        h = lstm.zero_state(batch_size, tf.float32)
        q_star = tf.zeros([batch_size, out_size], dtype=tf.float32)
        for step in range(time_steps):
            q, h = lstm.__call__(q_star, h)
            q = tf.reshape(q, [batch_size, in_size])
            q_ = tf.gather(q, node_numbers)
            e = tf.multiply(x, q_)
            e = tf.reduce_sum(e, axis=-1, keepdims=True)
            a = segment_softmax(e, node_numbers, num_node=batch_size, name='Attentions')
            r = tf.math.segment_sum(tf.multiply(a,x), node_numbers)
            q_star = tf.concat([q, r], axis=-1)
        return q_star