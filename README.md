# ccgnet
CCGNet (Co-Crystal Graph Network) is a deep learning framework for virtual screening of binary organic cocrystals, which integrates the priori knowledge into the feature learning on molecular graph and achieves a great improvement of performance for data-driven cocrystal screening.

Requirements:
* Python 3.7
* Tensorflow (1.6=<Version<2.0)
* RDKit
* Openbabel 2.4.1
* CCDC Python API 
* Scikit-learn

**Note**: if you don't have CCDC Python API, you can try the ccgnet-OB which has been removed this dependency for better open source, but the performance may decrease slightly.

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
~~~
from Featurize import Coformer

c = Coformer('./Test/coformers/1983.sdf')
A = c.AdjacentTensor.OnlyCovalentBond(with_coo=False)
A.shape
(20, 4, 20)
~~~
If with_coo=True, it will return the adjacent tensor with COO format, which is easy to feed other GNN framework, such as pytorch-geometric.
The method 'OnlyCovalentBond' only consider the four type of covalent bonds, such as single, double, triple and aromatic bonds.
If you want to add bond lenth:
~~~
A = c.AdjacentTensor.WithBondLenth(with_coo=False)
A.shape
#Out: (20, 5, 20)
~~~
The extra channel in axis=1 denotes the bond lenth. Each element either 0 or the bond lenth.
There are some other methods to compute adjacent tensor, please see Featurize/AdjacentTensor.py .

The atom features and global_state can be computed by:
~~~
V = c.VertexMatrix.feature_matrix()
global_state = c.descriptors()
print(V.shape, global_state.shape)
#Out: (20, 34) (12,)
~~~
Two Coformer objects can be transformed to a Cocrystal object. The features can be calculated by the same way:
~~~
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
~~~
A_hb = cc.CCGraphTensor(t_type='OnlyCovalentBond', hbond=True)
A_hb.shape
#Out: (34, 5, 34) 
~~~
If you want to add possible π-π stack:
~~~
A_hb_pp = cc.CCGraphTensor(t_type='OnlyCovalentBond', hbond=True, pipi_stack=True)
A_hb_pp.shape
#Out: (34, 6, 34)
#You can add edge between potential aromatic atoms, but the complexity of cc graph will increase.
~~~
Furthermore, if you want to add possible weak H-bond interaction, such as C-H···O, C-H···N :
~~~
A_hb_pp_c = cc.CCGraphTensor(t_type='OnlyCovalentBond', hbond=True, pipi_stack=True, contact=True)
A_hb_pp_c.shape
#Out: (34, 7, 34)
~~~
If you want to make your own dataset, ccgnet also provide Dataset class.
First, you need prepare two files like 'CC_Table.tab' and 'Mol_Blocks.dir' in './Samples/'.
'CC_Table.tab' reprensents the pairs of coformers. 
'Mol_Blocks.dir' is a text file with python dictionary format, whose key is the coformer name in 'CC_Table.tab' and value is string with 3D 'sdf' format.
Note that the input 'sdf' string should preferably be an optimized 3D structure.
~~~
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
We can build the dataset based on the two file:
~~~
from ccgnet.Dataset import Dataset, DataLoader

data = Dataset('./Samples/CC_Table.tab', mol_blocks_dir='./Samples/Mol_Blocks.dir')
data.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True, save_name=None)
~~~
If save_name is not None, the dataset will be saved as a 'save_name'.npz file to your disk. 
~~~
import tensorflow as tf
from ccgnet import experiment as exp
from ccgnet import layers
import time
from ccgnet.Dataset import Dataset, DataLoader
~~~
