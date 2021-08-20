import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from ccgnet.Dataset import Dataset, DataLoader
import numpy as np
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
import hyperopt.pyll.stochastic
import random


fold_10 = eval(open('Fold_10.dir').read())
def make_dataset():
    data1 = Dataset('CC_Table/CC_Table.tab', mol_blocks_dir='Mol_Blocks.dir')
    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
    return data1

def black_box_function(args_dict):
    clf = SVC(**args_dict, probability=0)
    clf.fit(x_train, y_train)
    valid_pred_labels = clf.predict(x_valid)
    valid_acc = accuracy_score(y_valid, valid_pred_labels)
    print((str(valid_acc)+': {}').format(args_dict))
    return {'loss': 1-valid_acc, 'status': STATUS_OK}

space4svc = {
                'C': hp.uniform('C', 0, 20),
                'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
                'gamma': hp.choice('gamma', ['scale','auto', hp.uniform('gamma-1', 0, 1)]),
                'degree': hp.choice('degree', range(1, 21)),
                'coef0': hp.uniform('coef0', 0, 10)
            }

Samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
random.seed(10)
random.shuffle(Samples)
num_sample = len(Samples)
train_num = int(0.9 * num_sample)
train_samples = Samples[:train_num]
valid_samples = Samples[train_num:]

data = make_dataset()

train_data, valid_data = data.split(train_samples=train_samples, valid_samples=valid_samples)
x_train, y_train = train_data[-2], train_data[2]
x_valid, y_valid = valid_data[-2], valid_data[2]

trials = Trials()
best = fmin(black_box_function, space4svc, algo=tpe.suggest, max_evals=100, trials=trials)
print(best)
