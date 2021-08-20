import sys
sys.path.append("../")
import glob
from sklearn.metrics import confusion_matrix, precision_score, recall_score, balanced_accuracy_score, f1_score, accuracy_score
import numpy as np
from Featurize import Coformer


def argmax(pred):
    return pred.index(max(pred))

class ParseTestLog(object):
    
    def __init__(self, logfile):
        fr = open(logfile).read().split('\n')
        self.labels = eval(fr[1])
        self.pred_score = eval(fr[2])
        self.pred_labels = [argmax(j) for j in self.pred_score]
        try:
            self.tags = [i.decode() for i in eval(fr[3])]
        except:
            self.tags = [i for i in eval(fr[3])]
        try:
            self.att_weigths = eval(fr[4])
        except:
            self.att_weigths = None
    
    def _print(self, metric_name, value):
        print('{}: {:.2f}'.format(metric_name, value*100))
    @property
    def Reports(self):
        self._print('Balanced Accuracy',self.bacc)
        self._print('Negative Accuracy',self.nacc)
        self._print('Positive Accuracy',self.pacc)
        self._print('Accuracy',self.acc)
        self._print('Precision',self.precision)
        self._print('Recall',self.recall)
        self._print('F1 Score',self.f1)
    @property    
    def bacc(self):
        return balanced_accuracy_score(self.labels, self.pred_labels)
    @property
    def precision(self):
        return precision_score(self.labels, self.pred_labels)
    @property
    def confusion_matrix(self):
        return confusion_matrix(self.labels, self.pred_labels)
    @property
    def recall(self):
        return recall_score(self.labels, self.pred_labels)
    @property
    def f1(self):
        return f1_score(self.labels, self.pred_labels)
    @property
    def nacc(self):
        cm = self.confusion_matrix
        return float(cm[0][0])/sum(cm[0])
    @property
    def pacc(self):
        cm = self.confusion_matrix
        return float(cm[1][1])/sum(cm[1])
    @property
    def acc(self):
        return accuracy_score(self.labels, self.pred_labels)
    @property
    def SortPredictScore(self):
        pos_all_score = np.array(self.pred_score)[:,1]
        sorted_ix = np.argsort(pos_all_score)[::-1]
        sorted_tags = np.array(self.tags)[sorted_ix].tolist()
        sorted_score = pos_all_score[sorted_ix].tolist()
        return list(zip(sorted_tags, sorted_score))
    
    def split_atts(self, cc_table_dir, mol_blocks='../data/Mol_Blocks.dir'):
        if self.att_weigths == None:
            raise ValueError('There are no attention weights in the test log.')
        cc_dic = get_info(cc_table_dir)
        mol_blocks = eval(open(mol_blocks).read())
        counter = 0
        info = {}
        for i in self.tags:
            info.setdefault(i, {})
            info[i]['coformers'] = [cc_dic[i][0], cc_dic[i][1]]
            c1 = mol_blocks[cc_dic[i][0]]
            c2 = mol_blocks[cc_dic[i][1]]
            c1 = Coformer(c1)
            c2 = Coformer(c2)
            n1, n2 = c1.atom_number, c2.atom_number
            info[i]['atom_number'] = [c1.atom_number, c2.atom_number]
            info[i]['attentions'] = self.att_weigths[counter:counter+n1+n2].tolist()
            counter = counter + n1+n2
        return info


class ParseTestLogEnsemble(object):
    def __init__(self, logset):
        self.logset = logset
        self.labels = [i.labels for i in logset]
        self.pred_score = [i.pred_score for i in logset]
        self.pred_labels = [i.pred_labels for i in logset]
        self.tags = [i.tags for i in logset]
        try:
            self.att_weigths = [i.att_weigths for i in logset]
        except:
            pass
        self.bagging_print = True
    
    def _print(self, metric_name, mean, std):
        print('{}: {:.2f}(+-{:.2f})'.format(metric_name, mean*100, std*100))
    @property        
    def Reports(self):
        self._print('Balanced Accuracy',self.bacc[0], self.bacc[1])
        self._print('Negative Accuracy',self.nacc[0], self.nacc[1])
        self._print('Positive Accuracy',self.pacc[0], self.pacc[1])
        self._print('Accuracy',self.acc[0], self.acc[1])
        self._print('Precision',self.precision[0], self.precision[1])
        self._print('Recall',self.recall[0], self.recall[1])
        self._print('F1 Score',self.f1[0], self.f1[1])
    @property
    def bacc(self):
        bacc = np.array([i.bacc for i in self.logset])
        mean = bacc.mean()
        std = bacc.std()
        return mean, std
    @property
    def precision(self):
        prec = np.array([i.precision for i in self.logset])
        mean = prec.mean()
        std = prec.std()
        return mean, std
    @property
    def confusion_matrix(self): 
        return [i.confusion_matrix for i in self.logset]
    @property
    def recall(self):
        rc = np.array([i.recall for i in self.logset])
        mean = rc.mean()
        std = rc.std()
        return mean, std
    @property
    def f1(self):
        f = np.array([i.f1 for i in self.logset])
        mean = f.mean()
        std = f.std()
        return mean, std
    @property
    def nacc(self):
        Nacc = np.array([i.nacc for i in self.logset])
        mean = Nacc.mean()
        std = Nacc.std()
        return mean, std
    @property
    def pacc(self):
        Pacc = np.array([i.pacc for i in self.logset])
        mean = Pacc.mean()
        std = Pacc.std()
        return mean, std
    @property
    def acc(self):
        Acc = np.array([i.acc for i in self.logset])
        mean = Acc.mean()
        std = Acc.std()
        return mean, std
    @property
    def Bagging(self):
        num_clf = len(self.logset)
        #all_labels = np.array(self.labels)[0]
        all_pred_labels = np.array(self.pred_labels).sum(axis=0)
        bagging = all_pred_labels > (num_clf/2)
        bagging = bagging.astype(int)
        bacc = balanced_accuracy_score(self.labels[0], bagging)
        tp = precision_score(self.labels[0], bagging)
        cm = confusion_matrix(self.labels[0], bagging)
        recall = recall_score(self.labels[0], bagging)
        f1 = f1_score(self.labels[0], bagging)
        nacc = float(cm[0][0])/sum(cm[0])
        pacc = float(cm[1][1])/sum(cm[1])
        if self.bagging_print:
            print('BACC: {:.2f}'.format(bacc*100))
            print('TP: {:.2f}'.format(tp*100))
            print('Nacc: {:.2f}'.format(nacc*100))
            print('Pacc: {:.2f}'.format(pacc*100))
            print('Recall: {:.2f}'.format(recall*100))
            print('F1: {:.2f}'.format(f1*100))
        return bagging
    @property
    def SortPredictScore(self):
        pos_all_score = np.array(self.pred_score).sum(axis=0)[:,1]
        neg_all_score = np.array(self.pred_score).sum(axis=0)[:,0]
        sorted_ix = np.argsort(pos_all_score)[::-1]
        sorted_tags = np.array(self.tags[0])[sorted_ix].tolist()
        #sorted_labels = np.array(self.labels[0])[sorted_ix]
        pos_sorted_score = pos_all_score[sorted_ix].tolist()
        neg_sorted_score = neg_all_score[sorted_ix].tolist()
        is_correct = np.array(['√' if i==self.labels[0][ix] else '×' for ix,i in enumerate(self.Bagging)])[sorted_ix]
        return list(zip(sorted_tags, neg_all_score, pos_sorted_score, is_correct))
    

def selectlatest(path):
    l = glob.glob(path+'/model*_info.txt')
    latest = sorted([(int(i.split('model-')[-1].split('_')[0]),i) for i in l])[-1][1]
    return latest

class ParseValidLog(object):
    
    def __init__(self, logpath):
        logfile = selectlatest(logpath)
        fr = open(logfile).read().split('\n')
        self.logfile = logfile
        self.labels = eval(fr[5])
        self.pred_score = eval(fr[6])
        self.pred_labels = [argmax(j) for j in self.pred_score]
        try:
            self.tags = [i.decode() for i in eval(fr[7])]
        except:
            self.tags = [i for i in eval(fr[7])]
        self.loss = float(fr[2].split(':')[-1])
        try:
            self.att_weigths = eval(fr[8])
        except:
            self.att_weigths = None
    
    def _print(self, metric_name, value):
        print('{}: {:.2f}'.format(metric_name, value*100))
    @property
    def Reports(self):
        self._print('Balanced Accuracy',self.bacc)
        self._print('Negative Accuracy',self.nacc)
        self._print('Positive Accuracy',self.pacc)
        self._print('Accuracy',self.acc)
        self._print('Precision',self.precision)
        self._print('Recall',self.recall)
        self._print('F1 Score',self.f1)
    @property    
    def bacc(self):
        return balanced_accuracy_score(self.labels, self.pred_labels)
    @property
    def precision(self):
        return precision_score(self.labels, self.pred_labels)
    @property
    def confusion_matrix(self):
        return confusion_matrix(self.labels, self.pred_labels)
    @property
    def recall(self):
        return recall_score(self.labels, self.pred_labels)
    @property
    def f1(self):
        return f1_score(self.labels, self.pred_labels)
    @property
    def nacc(self):
        cm = self.confusion_matrix
        return float(cm[0][0])/sum(cm[0])
    @property
    def pacc(self):
        cm = self.confusion_matrix
        return float(cm[1][1])/sum(cm[1])
    @property
    def acc(self):
        return accuracy_score(self.labels, self.pred_labels)
    @property
    def SortPredictScore(self):
        pos_all_score = np.array(self.pred_score)[:,1]
        sorted_ix = np.argsort(pos_all_score)[::-1]
        sorted_tags = np.array(self.tags)[sorted_ix].tolist()
        sorted_score = pos_all_score[sorted_ix].tolist()
        return zip(sorted_tags, sorted_score)
    
    def split_atts(self, cc_table_dir, mol_blocks='../data/Mol_Blocks.dir'):
        if self.att_weigths == None:
            raise ValueError('There are no attention weights in the test log.')
        cc_dic = get_info(cc_table_dir)
        mol_blocks = eval(open(mol_blocks).read())
        counter = 0
        info = {}
        for i in self.tags:
            info.setdefault(i, {})
            info[i]['coformers'] = [cc_dic[i][0], cc_dic[i][1]]
            c1 = mol_blocks[cc_dic[i][0]]
            c2 = mol_blocks[cc_dic[i][1]]
            c1 = Coformer(c1)
            c2 = Coformer(c2)
            n1, n2 = c1.atom_number, c2.atom_number
            info[i]['atom_number'] = [c1.atom_number, c2.atom_number]
            info[i]['attentions'] = self.att_weigths[counter:counter+n1+n2].tolist()
            counter = counter + n1+n2
        return info

def TestAccForEachMol(sample_list, log_list, is_return=False, is_print=True):
    #TEST_LIST = eval(open('test_list').read())
    #CL20 = eval(open('CL-20_Test.list').read())
    BACC, NACC, PACC = [], [], []
    L = []
    for log_ in log_list:
        log = ParseTestLog(log_)
        l = []
        L.append(log)
        for ix,i in enumerate(log.tags):
            if i in sample_list:
                l.append([ix, i])
        ix = np.array(np.array(l)[:,0], dtype=np.int32)
        #tags = np.array(np.array(l))[:,1]
        pred_labels = np.array(log.pred_labels)[ix]
        labels = np.array(log.labels)[ix]
        cm = confusion_matrix(labels, pred_labels)
        nacc = float(cm[0][0])/sum(cm[0])
        pacc = float(cm[1][1])/sum(cm[1])
        bacc = balanced_accuracy_score(labels, pred_labels)
        BACC.append(bacc)
        NACC.append(nacc)
        PACC.append(pacc)
    if is_print:
        print('######### Mean ########')
        print('{}: {:.2f}(±{:.2f})'.format('BACC',np.array(BACC).mean()*100,np.array(BACC).std()*100))
        print('{}: {:.2f}(±{:.2f})'.format('PACC',np.array(PACC).mean()*100,np.array(PACC).std()*100))
        print('{}: {:.2f}(±{:.2f})'.format('NACC',np.array(NACC).mean()*100,np.array(NACC).std()*100))
    ens = ParseTestLogEnsemble(L)
    ens.bagging_print = False
    tags = np.array(ens.tags[0])[ix]
    score = np.array(ens.pred_score).sum(axis=0)
    neg_score, pos_score = score[:,0], score[:,1]
    bacc_bagging = balanced_accuracy_score(labels,ens.Bagging[ix])
    bagging_cm = confusion_matrix(labels,ens.Bagging[ix])
    nacc_bagging = float(bagging_cm[0][0])/sum(bagging_cm[0])
    pacc_bagging = float(bagging_cm[1][1])/sum(bagging_cm[1])
    if is_print:
        print('######### Bagging ########')
        print('{}: {:.2f}'.format('TPR', pacc_bagging*100))
        print('{}: {:.2f}'.format('TNR', nacc_bagging*100))
        print('{}: {:.2f}'.format('BACC', bacc_bagging*100))
    if is_return:
        out = []
        #for index,tag in enumerate(tags):
            #if labels[index] == ens.Bagging[ix][index]:
                #out.append((tag, '√', neg_score[index], pos_score[index]))
            #else:
                #out.append((tag, '×', neg_score[index], pos_score[index]))
        
        for ix,items in enumerate(ens.SortPredictScore):
            if items[0] in sample_list:
                out.append(items)
        return out, {'BACC':bacc_bagging*100,'TPR':pacc_bagging*100,'TNR':nacc_bagging*100}
    
from rdkit import Chem
import openbabel as ob
import pybel
import pubchempy as pcp

def get_info(cc_table_dir):
    try:
        cc_dic = {}
        for items in eval(open(cc_table_dir).read()):
            cc_dic.setdefault(items[-1], items)
    except:
        cc_dic = {}
        for line in open(cc_table_dir).readlines():
            items = line.strip().split('\t')
            cc_dic.setdefault(items[-1], items)
    return cc_dic

def OutputSortedScore(pred_sorted, cc_table, mol_block, bagging_result=None, ecc_info=None):
    if bagging_result:
        bagging_dic = {i[0]:i[2] for i in bagging_result}
    else:
        bagging_dic = None
    cc_info = get_info(cc_table)
    mol_block = eval(open(mol_block).read())
    table = []
    for i in pred_sorted:
        try:
            items = ecc_info[i[0]]
        except:
            items = cc_info[i[0]]
        mb1 = mol_block[items[0]]
        mb2 = mol_block[items[1]]
        c1 = Chem.MolFromMolBlock(mb1)
        if c1 == None:
            obc1 = pybel.readstring('sdf',mb1)
            #c1 = Chem.MolFromMolBlock(obc1.write('mol'))
            smi1 = obc1.write('smi')
        else:
            smi1 = Chem.MolToSmiles(c1)
        c2 = Chem.MolFromMolBlock(mb2)
        if c2 == None:
            obc2 = pybel.readstring('sdf',mb2)
            #c2 = Chem.MolFromMolBlock(obc2.write('mol'))
            smi2 = obc2.write('smi')
        else:
            smi2 = Chem.MolToSmiles(c2)
        print('#############################')
        if bagging_dic:
            print(i[0], 'Score: {:.2f}'.format(i[1]), bagging_dic[i[0]])
            #table.append('\t'.join([name1, smi1, name2, smi2, i[0], '{:.2f}'.format(i[1]), bagging_dic[i[0]]]))
            table.append('\t'.join([smi1, smi2, i[0], '{:.2f}'.format(i[1]), bagging_dic[i[0]]]))
        else:
            print(i[0], 'Score: {:.2f}'.format(i[1]), i[-1])
            #table.append('\t'.join([name1, smi1, name2, smi2, i[0], '{:.2f}'.format(i[1]), i[-1]]))
            table.append('\t'.join([smi1, smi2, i[0], '{:.2f}'.format(i[1]), i[-1]]))
        #print(name1, smi1)
        #print(name2, smi2)
    return table