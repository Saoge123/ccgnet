import glob
from sklearn.metrics import confusion_matrix, precision_score, recall_score, balanced_accuracy_score, f1_score
import numpy as np
import pandas as pd
from decimal import Decimal 

def func(pred):
    return pred.index(max(pred))

def fix(d, tofixed=4):
    s = '00.'
    for i in range(tofixed):
        s = s + '0'
    decimal_string = str(Decimal(d).quantize(Decimal(s)))
    n_decimal_fractions = len(decimal_string.split('.')[-1])
    if n_decimal_fractions < tofixed:
        for i in range(tofixed-n_decimal_fractions):
            decimal_string = decimal_string + '0'
    return decimal_string

def model_metrics_report(path, is_print=False, tofixed=4, DNN_info=False):
    models = glob.glob(path+'/*')
    stat = {}
    for model in models:
        model_name = model.split('/')[-1]
        stat.setdefault(model_name, {})
        folds = glob.glob(model+'/*')
        stat[model_name]['TN_number'] = []
        stat[model_name]['TP_number'] = []
        for fold in folds:
            time = fold.split('/')[-1]
            stat[model_name].setdefault(time, {})
           
            infos = glob.glob(fold+'/model*_info.txt')
            d = {}
            for info in infos:
                key = int(info.split('_info')[0].split('-')[-1])
                d.setdefault(key, info)
            fr = open(d[max(d.keys())]).readlines()
            if 'bacc' in fr[2]:
                labels = eval(fr[6])
            else:
                labels = eval(fr[5])
            if DNN_info:
                labels = eval(fr[-1])
            n = 0
            for i in labels:
                if i == 0:
                    n+=1
            stat[model_name]['TN_number'].append(n)
            stat[model_name]['TP_number'].append(len(labels)-n)
            if 'bacc' in fr[2]:
                pred = [func(i) for i in eval(fr[7])]
            else:
                pred = [func(i) for i in eval(fr[6])]
            if DNN_info:
                pred = [func(i) for i in eval(fr[-2])]
            stat[model_name][time]['cm'] = confusion_matrix(labels, pred).tolist()
            #print(stat[model_name][time]['cm'])
            if sum(stat[model_name][time]['cm'][0]) == n:
                #print(n)
                stat[model_name][time]['Nacc'] = float(stat[model_name][time]['cm'][0][0])/sum(stat[model_name][time]['cm'][0])
                stat[model_name][time]['Pacc'] = float(stat[model_name][time]['cm'][1][1])/sum(stat[model_name][time]['cm'][1])
            else:
                stat[model_name][time]['Nacc'] = float(stat[model_name][time]['cm'][1][1])/sum(stat[model_name][time]['cm'][1])
                stat[model_name][time]['Pacc'] = float(stat[model_name][time]['cm'][0][0])/sum(stat[model_name][time]['cm'][0])
            stat[model_name][time]['tp'] = precision_score(labels, pred)
            stat[model_name][time]['recall'] = recall_score(labels, pred)
            stat[model_name][time]['f1_score'] = f1_score(labels, pred)
            stat[model_name][time]['bacc'] = balanced_accuracy_score(labels, pred)
            stat[model_name][time]['test_acc'] = float(fr[1].split(':')[-1].strip())
            stat[model_name][time]['train_acc'] = float(fr[3].split(':')[-1].strip())
    result = {}
    for key in stat:
        bacc = []
        recall = []
        tp = []
        test_acc = []
        train_acc = []
        f1 = []
        Nacc = []
        Pacc = []
        for k in stat[key]:
            if k in {'TP_number','TN_number'}:
                continue
            bacc.append(stat[key][k]['bacc'])
            recall.append(stat[key][k]['recall'])
            tp.append(stat[key][k]['tp'])
            test_acc.append(stat[key][k]['test_acc'])
            train_acc.append(stat[key][k]['train_acc'])
            f1.append(stat[key][k]['f1_score'])
            Nacc.append(stat[key][k]['Nacc'])
            Pacc.append(stat[key][k]['Pacc'])
        result[key] = {}
        result[key]['TP_number in each fold'] = ','.join([str(i) for i in stat[model_name]['TP_number']])
        result[key]['TN_number in each fold'] = ','.join([str(i) for i in stat[model_name]['TN_number']])
        result[key]['Train Accuracy'] = '{}(±{})'.format(fix(np.array(train_acc).mean()*100,tofixed=tofixed), 
                                                          fix(np.array(train_acc).std()*100, tofixed=tofixed))
        result[key]['Test Accuracy'] = '{}(±{})'.format(fix(np.array(test_acc).mean()*100,tofixed=tofixed), 
                                                                 fix(np.array(test_acc).std()*100,tofixed=tofixed))
        result[key]['Balanced Accuracy'] = '{}(±{})'.format(fix(np.array(bacc).mean()*100, tofixed=tofixed),
                                                                     fix(np.array(bacc).std()*100,tofixed=tofixed))
        result[key]['Nacc'] = '{}(±{})'.format(fix(np.array(Nacc).mean()*100,tofixed=tofixed), 
                                                      fix(np.array(Nacc).std()*100,tofixed=tofixed))
        result[key]['Pacc'] = '{}(±{})'.format(fix(np.array(Pacc).mean()*100,tofixed=tofixed), 
                                                      fix(np.array(Pacc).std()*100,tofixed=tofixed))
        result[key]['Precision'] = '{}(±{})'.format(fix(np.array(tp).mean()*100,tofixed=tofixed), 
                                                             fix(np.array(tp).std()*100,tofixed=tofixed))
        result[key]['Recall'] = '{}(±{})'.format(fix(np.array(recall).mean()*100,tofixed=tofixed), 
                                                          fix(np.array(recall).std()*100,tofixed=tofixed))
        result[key]['f1_score'] = '{}(±{})'.format(fix(np.array(f1).mean()*100,tofixed=tofixed), 
                                                            fix(np.array(f1).std()*100,tofixed=tofixed))
        if is_print:
            print(key)
            print('  TP_number in each fold: {}'.format(stat[model_name]['TP_number']))
            print('  TN_number in each fold: {}'.format(stat[model_name]['TN_number']))
            print('  Train Accuracy(ACC): {:.4f}(±{:.4f})'.format(np.array(train_acc).mean()*100, np.array(train_acc).std()*100))
            print('  Test Accuracy(ACC): {:.4f}(±{:.4f})'.format(np.array(test_acc).mean()*100, np.array(test_acc).std()*100))
            print('  Balanced Accuracy(BACC): {:.4f}(±{:.4f})'.format(np.array(bacc).mean()*100, np.array(bacc).std()*100))
            print('  Nacc: {:.4f}(±{:.4f})'.format(np.array(Nacc).mean()*100, np.array(Nacc).std()*100))
            print('  Pacc: {:.4f}(±{:.4f})'.format(np.array(Pacc).mean()*100, np.array(Nacc).std()*100))
            print('  Precision(True Positive, TP): {:.4f}(±{:.4f})'.format(np.array(tp).mean()*100, np.array(tp).std()*100))
            print('  Rcall: {:.4f}(±{:.4f})'.format(np.array(recall).mean()*100, np.array(recall).std()*100))
            print('  f1_score: {:.4f}(±{:.4f})'.format(np.array(f1).mean()*100, np.array(f1).std()*100))
    '''
    if is_return:
        return result
    '''
    return pd.DataFrame(result).T