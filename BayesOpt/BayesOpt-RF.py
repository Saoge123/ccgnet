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
    clf = RandomForestClassifier(**args_dict, n_jobs=8)
    clf.fit(x_train, y_train)
    valid_pred_score = clf.predict_proba(x_valid)
    valid_pred_labels = [np.argmax(i) for i in valid_pred_score]
    valid_acc = accuracy_score(y_valid, valid_pred_labels)
    print((str(valid_acc)+': {}').format(args_dict))
    return {'loss': 1-valid_acc, 'status': STATUS_OK}

space4rf = {
            'max_depth': hp.choice('max_depth', range(1, 21)),
            'max_features': hp.choice('max_features', range(1,24)),
            'n_estimators': hp.choice('n_estimators', list(range(10, 501, 10))),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'min_samples_leaf': hp.choice('min_samples_leaf', list(range(1, 50))),
            'min_samples_split': hp.choice('min_samples_split', list(range(2, 201))),
            'bootstrap': hp.choice('bootstrap', [0, 1])
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
best = fmin(black_box_function, space4rf, algo=tpe.suggest, max_evals=100, trials=trials)
print(best)