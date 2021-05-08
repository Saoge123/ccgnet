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
print(A.shape, V.shape, global_state.shape)
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
