import tensorflow as tf
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import time
import os


def create_graph_placeholders(dataset, use_desc=True, with_tags=True, with_attention=True, use_subgraph=False):
    '''
    dataset: should be a sequence (list, tuple or array) whose order is [V, A, Labels, masks, graph size, tags, descriptors]
    '''
    placeholders = []
    V_shape = [None] + list(dataset[0].shape[1:])
    V = tf.compat.v1.placeholder(tf.as_dtype(dataset[0].dtype), shape=V_shape, name='V_input')
    placeholders.append(V)
    A_shape = [None] + list(dataset[1].shape[1:])
    A = tf.compat.v1.placeholder(tf.as_dtype(dataset[1].dtype), shape=A_shape, name='AdjMat_input')
    placeholders.append(A)
    labels_shape = [None]
    labels = tf.compat.v1.placeholder(tf.as_dtype(dataset[2].dtype), shape=labels_shape, name='labels_input')
    placeholders.append(labels)
    mask_shape = [None] + list(dataset[3].shape[1:])
    masks = tf.compat.v1.placeholder(tf.as_dtype(dataset[3].dtype), shape=mask_shape, name='masks_input')
    placeholders.append(masks)
    if with_attention:
        graph_size_shape = [None]
        graph_size = tf.compat.v1.placeholder(tf.as_dtype(dataset[4].dtype), shape=graph_size_shape, name='graph_size_input')
        placeholders.append(graph_size)
    if with_tags:
        tags_shape = [None]
        tags = tf.compat.v1.placeholder(tf.as_dtype(dataset[5].dtype), shape=tags_shape, name='tags_input')
        placeholders.append(tags)
    if use_desc:
        global_state_shape = [None] + list(dataset[6].shape[1:])
        global_state = tf.compat.v1.placeholder(tf.as_dtype(dataset[6].dtype), shape=global_state_shape, name='global_state_input')
        placeholders.append(global_state)
    if use_subgraph:
        subgraph_size_shape = [None, 2]
        subgraph_size = tf.compat.v1.placeholder(tf.as_dtype(dataset[7].dtype), shape=subgraph_size_shape, name='subgraph_size_input')
        placeholders.append(subgraph_size)
    return placeholders

def create_fc_placeholders(dataset):
    embedding_shape = [None] + list(dataset[0].shape[1:])
    embedding = tf.compat.v1.placeholder(tf.as_dtype(dataset[0].dtype), shape=embedding_shape, name='Mol_Embedding')
    labels_shape = [None]
    labels = tf.compat.v1.placeholder(tf.as_dtype(dataset[1].dtype), shape=labels_shape, name='labels_input')
    tags_shape = [None]
    tags = tf.compat.v1.placeholder(tf.as_dtype(dataset[2].dtype), shape=labels_shape, name='tags_input')
    try:
        desc_shape = [None] + list(dataset[3].shape[1:])
        desc = tf.compat.v1.placeholder(tf.as_dtype(dataset[3].dtype), shape=desc_shape, name='desc_input')
        return [embedding, labels, tags, desc]
    except:
        return [embedding, labels, tags]

def create_input_variable(inputs):
    variable_initialization = {}
    for i in range(len(inputs)):
        placeholder = tf.compat.v1.placeholder(tf.as_dtype(inputs[i].dtype), shape=inputs[i].shape)
        var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        variable_initialization[placeholder] = inputs[i]
        inputs[i] = var
    return inputs, variable_initialization

def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def make_feed_dict(placeholders, data_batch):
    feed_dict = {}
    for i in range(len(placeholders)):
        feed_dict.setdefault(placeholders[i], data_batch[i])
    return feed_dict

def create_loss_function(V, labels, is_training):
    with tf.compat.v1.variable_scope('loss') as scope:
        print('Creating loss function and summaries')
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=V, labels=labels), name='cross_entropy')
        correct_prediction = tf.cast(tf.equal(tf.argmax(V, 1), tf.cast(labels, tf.int64)), tf.float32, name='correct_prediction')
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
        max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
        max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
        max_acc = tf.cond(is_training, lambda: tf.compat.v1.assign(max_acc_train, tf.maximum(max_acc_train, accuracy)), lambda: tf.compat.v1.assign(max_acc_test, tf.maximum(max_acc_test, accuracy)))
        tf.compat.v1.add_to_collection('losses', cross_entropy)
        tf.compat.v1.summary.scalar('accuracy', accuracy)
        tf.compat.v1.summary.scalar('max_accuracy', max_acc)
        tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)
        reports = {}
        reports['accuracy'] = accuracy
        reports['max acc.'] = max_acc
        reports['cross_entropy'] = cross_entropy
    return tf.add_n(tf.compat.v1.get_collection('losses')), reports

def make_train_step(loss, global_step, optimizer='adam', starter_learning_rate=0.1, learning_rate_step=1000, learning_rate_exp=0.1, reports=None):
    if reports==None:
        reports = {}
    print('Preparing training')
    if len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)) > 0:
        loss += tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)      
    with tf.control_dependencies(update_ops):
        if optimizer == 'adam':
            train_step = tf.compat.v1.train.AdamOptimizer().minimize(loss, global_step=global_step, name='train_step')
        else:
            learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, learning_rate_step, learning_rate_exp, staircase=True)
            train_step = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step, name='train_step')
            reports['lr'] = learning_rate
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
    return train_step, reports

def make_batch(data, epoch, batch_size, with_shuffle=True, name=None):
    with tf.compat.v1.variable_scope(name, default_name='input_slice') as scope:
        inputs = []
        for i in data:
            ph = tf.compat.v1.placeholder(tf.as_dtype(i.dtype), shape=i.shape)
            inputs.append(ph)
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(tuple(inputs))
        if with_shuffle:
            dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat(epoch)
        else:
            dataset = dataset.batch(batch_size).repeat(epoch)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    return iterator, inputs

class Model(object):
    def __init__(self, model, train_data, valid_data, with_test=False, test_data=None,  build_fc=False, 
                 model_name='model', dataset_name='dataset', with_tags=True, use_desc=True, use_subgraph=False,
                 with_attention=True, snapshot_path='./snapshot/', summary_path='./summary/'):
        tf.compat.v1.reset_default_graph()
        self.train_data = train_data
        self.test_data = valid_data
        self.val = with_test
        if self.val:
            self.val_data = test_data
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=(), name='is_training')
        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        self.build_fc = build_fc
        if build_fc:
            self.inputs = create_fc_placeholders(train_data)
        else:
            self.inputs = create_graph_placeholders(train_data, use_desc=use_desc, 
                                              with_tags=with_tags, with_attention=with_attention, use_subgraph=use_subgraph)
        self.pred_out, self.labels = model.build_model(self.inputs, self.is_training, self.global_step)
        self.snapshot_path = snapshot_path+'/%s/%s/' % (model_name, dataset_name)
        self.test_summary_path = summary_path+'/%s/test/%s' %(model_name, dataset_name)
        self.train_summary_path = summary_path+'/%s/train/%s' %(model_name, dataset_name)
        self.is_finetuning = False
        

    def create_batch(self, num_epoch=100, train_batch_size=256, test_batch_size=None):
        self.train_batch_num_per_epoch = int(self.train_data[0].shape[0]/train_batch_size) + 1
        self.train_batch_iterator, self.train_batch_placeholders = make_batch(self.train_data, num_epoch, train_batch_size, name='train_batch')
        if test_batch_size == None:
            self.test_batch_num_per_epoch = 1
            self.test_batch_iterator, self.test_batch_placeholders = make_batch(self.test_data, num_epoch, self.test_data[0].shape[0], with_shuffle=0, name='test_batch')
        else:
            self.test_batch_num_per_epoch = int(self.test_data[0].shape[0]/test_batch_size) + 1
            self.test_batch_iterator, self.test_batch_placeholders = make_batch(self.test_data, num_epoch, test_batch_size, with_shuffle=0, name='test_batch')
        if self.val:
            self.val_batch_iterator, self.val_batch_placeholders = make_batch(self.val_data, num_epoch, self.val_data[0].shape[0], with_shuffle=0, name='val_batch')
    
    def create_loss_function(self):
        self.loss, self.reports = create_loss_function(self.pred_out, self.labels, self.is_training)
    
    def make_train_step(self, optimizer='adam'):
        self.train_step, self.reports = make_train_step(self.loss, self.global_step, reports=self.reports, optimizer=optimizer)
    
    def fit(self, num_epoch=100, 
                  train_batch_size=256, 
                  test_batch_size=None, 
                  save_info=False, 
                  save_history=True, 
                  save_model=True,
                  save_att=False, 
                  metric='acc',  # one of the ['bacc','acc','loss']
                  silence=False, 
                  optimizer='adam', 
                  save_summary=True, 
                  early_stop=False,
                  early_stop_cutoff=20, 
                  max_to_keep=5):
        '''
        
        '''
        self.create_batch(num_epoch=num_epoch, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        self.create_loss_function()
        self.make_train_step(optimizer=optimizer)
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            ####################### initialization ########################
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(self.train_batch_iterator.initializer, feed_dict=make_feed_dict(self.train_batch_placeholders, self.train_data))
            sess.run(self.test_batch_iterator.initializer, feed_dict=make_feed_dict(self.test_batch_placeholders, self.test_data))
            self.train_samples = self.train_batch_iterator.get_next()
            self.test_samples = self.test_batch_iterator.get_next()
            if self.val:
                sess.run(self.val_batch_iterator.initializer, feed_dict=make_feed_dict(self.val_batch_placeholders, self.val_data))
                self.val_samples = self.val_batch_iterator.get_next()
            sess.run(tf.compat.v1.local_variables_initializer())
            if self.is_finetuning:
                self.restore_saver.restore(sess, self.restore_file)
            ###################### Starting summaries #####################
            print('Starting summaries')
            test_writer = tf.compat.v1.summary.FileWriter(self.test_summary_path, sess.graph)
            train_writer = tf.compat.v1.summary.FileWriter(self.train_summary_path, sess.graph)
            summary_merged = tf.compat.v1.summary.merge_all()
            ###################### training record #########################
            self.test_max_acc = {}
            self.test_max_acc['valid_acc'] = []
            self.test_max_acc['valid_cross_entropy'] = []
            self.test_max_acc['train_acc'] = []
            self.test_max_acc['train_cross_entropy'] = []
            if metric == 'bacc':
                self.test_max_acc['valid_bacc'] = []
            ###################### configure model saver #######################
            var_list = [var for var in  tf.compat.v1.global_variables() if "moving" in var.name]
            var_list += [var for var in  tf.compat.v1.global_variables() if "Moving" in var.name]
            var_list +=  tf.compat.v1.trainable_variables()
            saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=max_to_keep)
            if self.build_fc:
                ix_of_label_for_saving = 1
                ix_of_tag_for_saving = 2
            else:
                ix_of_label_for_saving = 2
                ix_of_tag_for_saving = 5
            ####################################################################
            if save_att:
                # Saving the att coefficients of each atom for visualization
                graph = tf.compat.v1.get_default_graph()
                try:
                    att_op = graph.get_operation_by_name('Global_Attention/Attentions').outputs[0]
                except:
                    att_op = graph.get_operation_by_name('Multi_Head_Global_Attention/Attentions').outputs[0]
            ####################################################################
            test_metric_cutoff = float('inf') if metric=='loss' else 0.0
            early_stop_counter = 0
            try:
                for epo in range(num_epoch):
                    ####################### train ######################
                    train_acc = 0.0
                    train_loss = 0.0
                    start_time = time.time()
                    for b in range(self.train_batch_num_per_epoch):
                        train_batch = sess.run([self.train_samples])[0]
                        feed_dict = make_feed_dict(self.inputs, train_batch)
                        feed_dict[self.is_training] = 1
                        summary, _, train_reports = sess.run([summary_merged, self.train_step, self.reports], feed_dict=feed_dict)
                        train_acc += train_reports['accuracy']
                        train_loss += train_reports['cross_entropy']
                    if save_summary:
                        train_writer.add_summary(summary, epo)
                    self.test_max_acc['train_acc'].append(train_acc/self.train_batch_num_per_epoch)
                    self.test_max_acc['train_cross_entropy'].append(train_loss/self.train_batch_num_per_epoch)
                    ####################### test ######################
                    test_acc = 0.0
                    test_loss = 0.0
                    test_tags = []
                    test_labels = []
                    for b in range(self.test_batch_num_per_epoch):
                        test_batch = sess.run([self.test_samples])[0]
                        feed_dict = make_feed_dict(self.inputs, test_batch)
                        feed_dict[self.is_training] = 0
                        test_tags.append(test_batch[ix_of_tag_for_saving])
                        test_labels.append(test_batch[ix_of_label_for_saving])
                        if save_att:
                            summary, test_reports, out, att = sess.run([summary_merged, self.reports, self.pred_out, att_op], 
                                                                        feed_dict=feed_dict)
                        else:
                            summary, test_reports, out = sess.run([summary_merged, self.reports, self.pred_out], 
                                                                   feed_dict=feed_dict)
                        test_acc += test_reports['accuracy']
                        test_loss += test_reports['cross_entropy']
                    if save_summary:
                        test_writer.add_summary(summary, epo)
                    self.test_max_acc['valid_acc'].append(test_acc/self.test_batch_num_per_epoch)
                    self.test_max_acc['valid_cross_entropy'].append(test_loss/self.test_batch_num_per_epoch)
                    test_tags = np.concatenate(test_tags)
                    test_labels = np.concatenate(test_labels)
                    if metric == 'bacc':
                        out_label = np.array([np.where(j==np.max(j)) for j in out]).reshape(-1)
                        test_bacc = balanced_accuracy_score(test_labels, out_label)
                        self.test_max_acc['valid_bacc'].append(test_bacc)
                        save_metric = test_bacc
                        is_save = save_metric > test_metric_cutoff
                    elif metric == 'acc':
                        save_metric = self.test_max_acc['valid_acc'][-1]
                        is_save = save_metric > test_metric_cutoff
                    elif metric == 'loss':
                        save_metric = self.test_max_acc['valid_cross_entropy'][-1]
                        is_save = save_metric < test_metric_cutoff
                    else:
                        raise ValueError("metric should be the one of ['bacc','acc','loss']")
                    ###########################  save model ############################
                    if is_save:
                        early_stop_counter = 0
                        test_metric_cutoff = save_metric
                        L = str(test_labels.tolist())
                        out = str(out.tolist())
                        if save_model:
                            verify_dir_exists(self.snapshot_path)
                            saver.save(sess, self.snapshot_path+'TestAcc-{:.2f}'.format(save_metric*100), global_step=epo)
                        ###################### Validation ####################
                        if self.val:
                            val_batch = sess.run([self.val_samples])[0]
                            feed_dict = make_feed_dict(self.inputs, val_batch)
                            feed_dict[self.is_training] = 0
                            val_tags = val_batch[ix_of_tag_for_saving]
                            val_labels = val_batch[ix_of_label_for_saving]
                            if save_att:
                                val_reports, val_out, val_att = sess.run([self.reports, self.pred_out, att_op], feed_dict=feed_dict)
                            else:
                                val_reports, val_out = sess.run([self.reports, self.pred_out], feed_dict=feed_dict)
                            val_info = [str(val_reports['accuracy']), 
                                        str(val_labels.tolist()), 
                                        str(val_out.tolist()),
                                        str(val_tags.tolist())]
                            if save_att:
                                val_info.append(str(val_att.tolist()))
                            self.test_max_acc['test_acc'] = val_reports['accuracy']
                        ######################## save model information ########################
                        if save_info:
                            if save_att:
                                att = str(att.tolist())
                                model_info = ['step:{}'.format(epo),
                                              'valid_acc:{}'.format(test_reports['accuracy']),
                                              'valid_cross_entropy:{}'.format(test_reports['cross_entropy']),
                                              'train_acc:{}'.format(self.test_max_acc['train_acc'][epo]),
                                              'train_cross_entropy:{}'.format(self.test_max_acc['train_cross_entropy'][epo]),
                                              L, out, str(test_tags.tolist()), att]
                            else:
                                model_info = ['step:{}'.format(epo),
                                              'valid_acc:{}'.format(test_reports['accuracy']),
                                              'valid_cross_entropy:{}'.format(test_reports['cross_entropy']),
                                              'train_acc:{}'.format(self.test_max_acc['train_acc'][epo]),
                                              'train_cross_entropy:{}'.format(self.test_max_acc['train_cross_entropy'][epo]),
                                              L, out, str(test_tags.tolist())]
                            if metric == 'bacc':
                                model_info.insert(2,'valid_bacc:{}'.format(save_metric))
                            open(self.snapshot_path+'model-{}_info.txt'.format(epo), 'w').writelines('\n'.join(model_info))
                            if self.val:
                                open(self.snapshot_path+'model-val-info.txt', 'w').writelines('\n'.join(val_info))
                    else:
                        early_stop_counter += 1
                    end_time = time.time()
                    if silence == False:
                        elapsed_time = end_time - start_time
                        print_content = '## Epoch {} ==> Train Loss:{:.5f}, Train Acc:{:.2f}, Valid Loss:{:.5f}, Valid Acc:{:.2f}, Elapsed Time:{:.2f} s'
                        print_content = print_content.format(epo,
                                                             self.test_max_acc['train_cross_entropy'][epo],
                                                             self.test_max_acc['train_acc'][epo]*100,
                                                             self.test_max_acc['valid_cross_entropy'][epo],
                                                             self.test_max_acc['valid_acc'][epo]*100,
                                                             elapsed_time)
                        print(print_content)
                    if early_stop_counter == early_stop_cutoff and early_stop != False:
                        print('Early stopping ...')
                        break
            except tf.errors.OutOfRangeError:
                print("done")
            finally:
                if save_history:
                    verify_dir_exists(self.snapshot_path)
                    open(self.snapshot_path+'history.dir' ,'w').write(str(self.test_max_acc))
        sess.close()
        return self.test_max_acc
