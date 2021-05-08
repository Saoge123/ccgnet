import tensorflow as tf
#from .helper import *


def get_inputs(model_path):
    saver = tf.compat.v1.train.import_meta_graph(model_path)
    graph = tf.compat.v1.get_default_graph()
    placeholders = {}
    V = graph.get_tensor_by_name('V_input:0')
    placeholders.setdefault(V.name.split('_input')[0], V)
    A = graph.get_tensor_by_name('AdjMat_input:0')
    placeholders.setdefault('A', A)
    labels = graph.get_tensor_by_name('labels_input:0')
    placeholders.setdefault(labels.name.split('_input')[0], labels)
    masks = graph.get_tensor_by_name('masks_input:0')
    placeholders.setdefault(masks.name.split('_input')[0], masks)
    try:
        graph_size = graph.get_tensor_by_name('graph_size_input:0')
        placeholders.setdefault(graph_size.name.split('_input')[0], graph_size)
    except:
        print("The name 'graph_size_input:0' refers to a Tensor which does not exist.")
    try:
        tags = graph.get_tensor_by_name('tags_input:0')
        placeholders.setdefault(tags.name.split('_input')[0], tags)
    except:
        print("The name 'tags_input:0' refers to a Tensor which does not exist.")
    try:
        global_state = graph.get_tensor_by_name('global_state_input:0')
        placeholders.setdefault(global_state.name.split('_input')[0], global_state)
    except:
        print("The name 'global_state_input:0' refers to a Tensor which does not exist.")
    try:
        subgraph_size = graph.get_tensor_by_name('subgraph_size_input:0')
        placeholders.setdefault(subgraph_size.name.split('_input')[0], subgraph_size)
    except:
        print("The name 'subgraph_size_input:0' refers to a Tensor which does not exist.")
    return saver, graph, placeholders

def get_feed_dict(data, placeholders):
    feed_dict = {}
    for key in data:
        feed_dict.setdefault(placeholders[key], data[key])
    return feed_dict

def get_pred_result(pred, tags):
    pos_pred = []
    for ix, i in enumerate(pred):
        if i[0] <= i[1]:
            pos_pred.append((i[1], tags[ix]))
    pos_pred = sorted(pos_pred, reverse=True)
    content = '\n'.join(['\t'.join(i) for i in pos_pred])
    print(content)
    return pos_pred

class Inference(object):
    def __init__(self, meta_file_path, weight_file_path, data):
        '''
        meta_file_path: .meta file that holds the graph structure
        weight_file_path: weight file which contains parameters of model
        data: a dictionary whose keys is a subset of {'V','A','labels','masks','graph_size','tags','global_state'},
              and its values is numpy array.
        '''
        tf.reset_default_graph()
        self.saver, self.graph, self.placeholders = get_inputs(meta_file_path)
        self.weight_file_path = weight_file_path
        self.feed_dict = get_feed_dict(data, self.placeholders)
            
    def predict(self, with_inference=True, with_att=True):
        with tf.Session() as sess:
            self.saver.restore(sess, self.weight_file_path)
            accuracy = self.graph.get_tensor_by_name('loss/accuracy:0')
            out = graph.get_tensor_by_name('final/result:0')
            if with_att:
                try:
                    att_op = self.graph.get_operation_by_name('Global_Attention/Attentions').outputs[0]
                except:
                    att_op = self.graph.get_operation_by_name('Multi_Head_Global_Attention/Attentions').outputs[0]
            if with_inference:
                if with_att:
                    self.pred, self.att = sess.run([out,att_op], feed_dict=self.feed_dict)
                    return self.pred, self.att
                else:
                    self.pred = sess.run(out, feed_dict=self.feed_dict)
                    return self.pred
            else:
                if with_att:
                    self.acc, self.pred, self.att = sess.run([accuracy, out, att_op], feed_dict=self.feed_dict)
                    return self.acc, self.pred, self.att
                else:
                    self.acc, self.pred = sess.run([accuracy, out], feed_dict=self.feed_dict)
                    return self.acc, self.pred