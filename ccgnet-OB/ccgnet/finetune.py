import tensorflow as tf
from .experiment import Model


def select_vars(remove_var=['loss/max_acc_test:0','loss/max_acc_train:0'], remove_keywords=['FC', 'final'], global_finetuning=True):
    if remove_keywords != []:
        for keyword in remove_keywords:
            vs = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=keyword)
            remove_var = remove_var+[v.name for v in vs]
    trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    var_to_restore = []
    var_to_remove = []
    for var in trainable_vars:
        if var.name in remove_var:
            var_to_remove.append(var)
        else:
            var_to_restore.append(var)
    restore_saver = tf.compat.v1.train.Saver(var_to_restore)
    if global_finetuning:
        return restore_saver, tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) 
    else:
        return restore_saver, var_to_remove

class Finetuning(Model):
    def __init__(self, model, 
                       train_data, 
                       test_data,
                       restore_file=None, 
                       remove_var=[], 
                       remove_keywords=['FC', 'final'],
                       global_finetuning=True, 
                       **kwargs):
        super(Finetuning, self).__init__(model, train_data, test_data,**kwargs)
        self.is_finetuning = True
        self.global_finetuning = global_finetuning
        self.restore_file = restore_file
        remove_var = remove_var + ['loss/max_acc_test:0','loss/max_acc_train:0']
        self.restore_saver, self.trainable_vars = select_vars(remove_var=remove_var, 
                                                              remove_keywords=remove_keywords, 
                                                              global_finetuning=global_finetuning)
        
    def make_train_step(self, optimizer='adam', starter_learning_rate=0.1, learning_rate_step=1000, learning_rate_exp=0.1):
        if self.reports==None:
            self.reports = {}
        print('Preparing training')
        if len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            self.loss += tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        update_ops = tf.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)      
        with tf.control_dependencies(update_ops):
            if optimizer == 'adam':
                self.train_step = tf.compat.v1.train.AdamOptimizer().minimize(self.loss, 
                                                               global_step=self.global_step, 
                                                               var_list=self.trainable_vars, 
                                                               name='train_step')
            else:
                learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, 
                                                           self.global_step, 
                                                           learning_rate_step, 
                                                           learning_rate_exp, 
                                                           staircase=True)
                self.train_step = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, 
                                                                                     global_step=self.global_step, 
                                                                                     var_list=self.trainable_vars, 
                                                                                     name='train_step')
                self.reports['lr'] = learning_rate
                tf.summary.scalar('learning_rate', learning_rate)
        return self.train_step, self.reports