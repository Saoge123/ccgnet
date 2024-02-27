# Co-Crystal Graph Network (CCGNet)
![image](https://github.com/Saoge123/ccgnet/blob/main/img/cfg.png)

CCGNet (Co-Crystal Graph Network) is a deep learning framework for virtual screening of binary organic cocrystals, which integrates the priori knowledge into the feature learning on molecular graph and achieves a great improvement of performance for data-driven cocrystal screening. https://doi.org/10.1038/s41467-021-26226-7

Requirements:
* Python 3.7
* Tensorflow (1.6=<Version<2.0)
* RDKit
* Openbabel 2.4.1
* CCDC Python API 
* Scikit-learn

**Note**: CCDC Python API is a commercial module, which is included in CSD softwares.  

The ccgnet module could be used to build CCGNet models. In ccgnet, the message passing is implement by Graph-CNN, where the code of this part we borrow from https://github.com/fps7806/Graph-CNN. The Featurize module represents the combination of two molecules as the input of CCGNet models.
# usage
We integrated the ensemble model as pipeline that can provide the high throughput screening for the defined compounds pairs and generate a report form automatically. 
predict.py can run the ensemble model that were composed by trained 10 models, whose mean balanced accuracy achieve 98.60% in 10-fold cross validation.
~~~
usage: predict.py [-h] [-table TABLE] [-mol_dir MOL_DIR] [-out OUT]
                  [-fmt {sdf,mol,mol2}] [-model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -table TABLE          The table of coformer pairs
  -mol_dir MOL_DIR      path of molecular file, default is "./"
  -out OUT              name of predictive file, default is "Result.xlsx"
  -fmt {sdf,mol,mol2}   format of molecular file，which support sdf, mol, mol2.
                        default is sdf
  -model_path MODEL_PATH
                        The path where the models are stored, default is
                        ./snapshot/CCGNet_block/CC_Dataset/
~~~
First, you need to prepare the structure files of each coformer, whose format can be 'sdf', 'mol', 'mol2'.
Then, a table that reprensents the pairs of coformers should be generated. Each item in line is separated by "\t". like this:

![image](https://github.com/Saoge123/ccgnet/blob/main/img/table-example.png)

The value of the last column is arbitrary, as long as it is an integer. 

Here, we use the testing samples in our work as an example
~~~
python predict.py -table ./Test/Test_Table.tab -mol_dir ./Test/coformers -out cc_test.xlsx -fmt sdf -model_path ./inference_model/CC_Model/
~~~
All pairs were sorted by the predictve score from high to low. The higher score, the higher possibility to form cocrystal.

Finally, the predictive result is written in a .xlsx file, eg. cc_test.xlsx. 
![image](https://github.com/Saoge123/ccgnet/blob/main/img/xlsx.png)

In CCGNet, Coformer object is the basic data unit, for each compound:
~~~python
from Featurize import Coformer

c = Coformer('./Test/coformers/1983.sdf')
A = c.AdjacentTensor.OnlyCovalentBond(with_coo=False)
A.shape
(20, 4, 20)
~~~
If with_coo=True, it will return the adjacent tensor with COO format, which is easy to feed other GNN framework, such as pytorch-geometric.
The method 'OnlyCovalentBond' only consider the four type of covalent bonds, such as single, double, triple and aromatic bonds.
If you want to add bond lenth:
~~~python
A = c.AdjacentTensor.WithBondLenth(with_coo=False)
A.shape
#Out: (20, 5, 20)
~~~
The extra channel in axis=1 denotes the bond lenth. Each element either 0 or the bond lenth.
There are some other methods to compute adjacent tensor, please see Featurize/AdjacentTensor.py .

The atom features and global_state can be computed by:
~~~python
V = c.VertexMatrix.feature_matrix()
global_state = c.descriptors()
print(V.shape, global_state.shape)
#Out: (20, 34) (12,)
~~~
Two Coformer objects can be transformed to a Cocrystal object. The features can be calculated by the same way:
~~~python
from Featurize import Coformer, Cocrystal

c1 = Coformer('./Test/coformers/1983.sdf')
c2 = Coformer('./Test/coformers/1110.sdf')
cc = Cocrystal(c1, c2)
A_cc = cc.AdjacentTensor.OnlyCovalentBond(with_coo=False)
V_cc = cc.VertexMatrix.feature_matrix()
global_state_cc = cc.descriptors()
print(A_cc.shape, V_cc.shape, global_state_cc.shape)
#Out: (34, 4, 34) (34, 34) (24,)
~~~
Here, we also defined the possible intermolecular interaction. To reprensent potential H-bonds, you can add edge between potential H-bond donnors and accptors
~~~python
A_hb = cc.CCGraphTensor(t_type='OnlyCovalentBond', hbond=True)
A_hb.shape
#Out: (34, 5, 34) 
~~~
If you want to add possible π-π stack:
~~~python
A_hb_pp = cc.CCGraphTensor(t_type='OnlyCovalentBond', hbond=True, pipi_stack=True)
A_hb_pp.shape
#Out: (34, 6, 34)
#You can add edge between potential aromatic atoms, but the complexity of cc graph will increase.
~~~
Furthermore, if you want to add possible weak H-bond interaction, such as C-H···O, C-H···N :
~~~python
A_hb_pp_c = cc.CCGraphTensor(t_type='OnlyCovalentBond', hbond=True, pipi_stack=True, contact=True)
A_hb_pp_c.shape
#Out: (34, 7, 34)
~~~
If you want to make your own dataset, ccgnet also provide Dataset class.
First, you need prepare two files like 'CC_Table.tab' and 'Mol_Blocks.dir' in './Samples/'.
'CC_Table.tab' reprensents the pairs of coformers. 
'Mol_Blocks.dir' is a text file with python dictionary format, whose key is the coformer name in 'CC_Table.tab' and value is string with 3D 'sdf' format.

**Note** that the input 'sdf' string should preferably be an optimized 3D structure.
~~~python
mol_block = eval(open('./Samples/Mol_Blocks.dir').read())
print(mol_block['1983'])
'''
1983
  -OEChem-03111921513D

 20 20  0     0  0  0  0  0  0999 V2000
    3.8509    0.4516    0.0012 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5999    1.4041   -0.0018 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5705   -0.7171    0.0001 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2066   -0.4231   -0.0002 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2205    0.9047    0.0004 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7298   -1.4570   -0.0007 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5841    1.1986    0.0002 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0933   -1.1629   -0.0007 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5204    0.1648   -0.0003 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6485    0.1782    0.0009 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.9735   -0.5420    0.0010 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4436    1.7577    0.0012 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4113   -2.4963   -0.0010 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8010   -1.7086    0.0001 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9053    2.2370    0.0009 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.8180   -1.9726   -0.0008 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.0655   -1.1463   -0.9058 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.7904    0.1844    0.0288 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.0445   -1.1886    0.8802 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.9650    1.4176    0.0017 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  9  1  0  0  0  0
  1 20  1  0  0  0  0
  2 10  2  0  0  0  0
  3  4  1  0  0  0  0
  3 10  1  0  0  0  0
  3 14  1  0  0  0  0
  4  5  2  0  0  0  0
  4  6  1  0  0  0  0
  5  7  1  0  0  0  0
  5 12  1  0  0  0  0
  6  8  2  0  0  0  0
  6 13  1  0  0  0  0
  7  9  2  0  0  0  0
  7 15  1  0  0  0  0
  8  9  1  0  0  0  0
  8 16  1  0  0  0  0
 10 11  1  0  0  0  0
 11 17  1  0  0  0  0
 11 18  1  0  0  0  0
 11 19  1  0  0  0  0
M  END
$$$$
'''
~~~
We can build the dataset based on these two file, such as:
~~~python
from ccgnet.Dataset import Dataset, DataLoader

data1 = Dataset('data/CC_Table/ECC&CC_Table.tab', mol_blocks_dir='data/Mol_Blocks.dir')
data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
data2 = Dataset('data/CC_Table/CC_Table-DataAug.tab', mol_blocks_dir='data/Mol_Blocks.dir')
data2.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
~~~
If save_name is not None, the dataset will be saved as a 'save_name'.npz file to your disk.
If make_dataframe=True, object data will make a new property .dataframe. Each entry holds the data for the corresponding sample, like:
~~~python
data.dataframe['UNEYOB'].keys()
#Out: dict_keys(['V', 'A', 'label', 'global_state', 'tag', 'mask', 'graph_size', 'subgraph_size'])
~~~
To train and test the model, we firstly split out the test set.
~~~python
nico = eval(open('data/Test/Test_Samples/Nicotinamide_Test.list').read())
carb = eval(open('data/Test/Test_Samples/Carbamazepine_Test.list').read())
indo = eval(open('data/Test/Test_Samples/Indomethacin_Test.list').read())
para = eval(open('data/Test/Test_Samples/Paracetamol_Test.list').read())
pyre = eval(open('data/Test/Test_Samples/Pyrene_Test.list').read())
apis = list(set(nico + indo + para + carb))
cl20 = eval(open('data/Test/Test/Test_Samples/CL-20_Test.list').read())
tnt = eval(open('data/Test/Test_Samples/TNT_Test.list').read())
# these samples that are selected as out-of-sample test are the same with our paper.
test = list(set(nico + carb + indo + para + pyre))
~~~
Here, we perform a 10-fold cross-validation.
~~~python
from sklearn.model_selection import KFold

cv = np.array([i for i in data.dataframe if i not in test])
np.random.shuffle(cv)
fold_10 = {}
fold_10['test'] = test
kf = KFold(n_splits=10, shuffle=True, random_state=1000)
n = 0
for train_idx, test_idx in kf.split(cv):
    fold = 'fold-{}'.format(n)
    fold_10[fold] = {}
    fold_10[fold]['train'] = cv[train_idx]
    fold_10[fold]['valid'] = cv[test_idx]
    n += 1
~~~
The model construction.
~~~python
import tensorflow as tf
from ccgnet import experiment as exp
from ccgnet.finetune import *
from ccgnet import layers
from ccgnet.layers import *
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score
from ccgnet.Dataset import Dataset, DataLoader
from Featurize.Coformer import Coformer
from Featurize.Cocrystal import Cocrystal

class CCGNet(object):
    def build_model(self, inputs, is_training, global_step):
        V = inputs[0]
        A = inputs[1]
        labels = inputs[2]
        mask = inputs[3]
        graph_size = inputs[4]
        tags = inputs[5]
        global_state = inputs[6]
        subgraph_size = inputs[7]
        # message passing 
        V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=64, mask=mask, num_updates=global_step, is_training=is_training)
        V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=16, mask=mask, num_updates=global_step, is_training=is_training)
        V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=64, mask=mask, num_updates=global_step, is_training=is_training)
        V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=16, mask=mask, num_updates=global_step, is_training=is_training)
        # readout
        V = ReadoutFunction(V, global_state, graph_size, num_head=2, is_training=is_training)
        # predict
        with tf.compat.v1.variable_scope('Predictive_FC_1') as scope:
            V = layers.make_embedding_layer(V, 256)
            V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
            V = tf.nn.relu(V)
            V = tf.compat.v1.layers.dropout(V, 0.457, training=is_training)
        with tf.compat.v1.variable_scope('Predictive_FC_2') as scope:
            V = layers.make_embedding_layer(V, 1024)
            V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
            V = tf.nn.relu(V)
            V = tf.compat.v1.layers.dropout(V, 0.457, training=is_training)
        with tf.compat.v1.variable_scope('Predictive_FC_3') as scope:
            V = layers.make_embedding_layer(V, 256)
            V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
            V = tf.nn.relu(V)
            V = tf.compat.v1.layers.dropout(V, 0.457, training=is_training)
        out = layers.make_embedding_layer(V, 2, name='final')
        return out, labels
~~~
Then, we will fit model. In this case, model and log will be saved at './snapshot/CCGNet_block/CC_Dataset/time_*'.

We call data.split to split train_data, valid_data and test_data. The test_data is optional. In model training, ccgnet will save the models that hold best 5 performance (default). You can increase or decrease the number of saving models by changing the value of 'max_to_keep'. When higher performance is achieved in valid set, ccgnet will save the weights and make inference for test set (if with_test=True). Also, you can use the saved model to perform inference on the test set by using predict.py.
~~~python
start = time.time()
snapshot_path = './snapshot/'
model_name = 'CCGNet'
dataset_name = 'CC_Dataset'
for fold in ['fold-{}'.format(i) for i in range(10)]:
    print('\n################ {} ################'.format(fold))
    train_data1, valid_data1, test_data1 = data1.split(train_samples=fold_10[fold]['train'], valid_samples=fold_10[fold]['valid'], with_test=True, test_samples=fold_10['test'])
    train_data2, valid_data2 = data2.split(train_samples=fold_10[fold]['train'], valid_samples=fold_10[fold]['valid'], with_test=False)
    train_data = []
    for ix, i in enumerate(train_data1):
        train_data.append(np.concatenate([i, train_data2[ix]]))
    del train_data2
    del train_data1
    tf.reset_default_graph()
    model = CCGNet()
    model = exp.Model(model, train_data, valid_data1, with_test=True, test_data=test_data1, snapshot_path=snapshot_path, use_subgraph=True,
                      model_name=model_name, dataset_name=dataset_name+'/time_{}'.format(fold[-1]))
    history = model.fit(num_epoch=100, save_info=True, save_att=True, silence=0, train_batch_size=128,
                        metric='acc')
end = time.time()
time_gap = end-start
h = time_gap//3600
h_m = time_gap%3600
m = h_m//60
s = h_m%60
print('{}h {}m {}s'.format(int(h),int(m),round(s,2)))
~~~
You can use the Featurize.MetricsReport module to assess model performance in 10-fold CV.
~~~python
from Featurize.MetricsReport import model_metrics_report

model_metrics_report('{}/{}/'.format(snapshot_path, model_name))
~~~
