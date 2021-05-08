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
Then, a table that reprensents the pairs of coformers should be generated. Each item in line is separated by "tab". like this:

![image](https://github.com/Saoge123/ccgnet/blob/main/img/table-example.png)

The value of the last column is arbitrary, as long as it is an integer. 

e.g. We use the test samples
~~~
python predict.py -table ./Test/Test_Table.tab -mol_dir ./Test/coformers -out cc_test.xlsx -fmt sdf -model_path ./inference_model/CC_Model/
~~~
All pairs were sorted by the predictve score from high to low. The higher score, the higher possibility to form cocrystal.

Finally, the predictive result is written in a .xlsx file, eg. cc_test.xlsx. 
![image](https://github.com/Saoge123/ccgnet/blob/main/img/xlsx.png)
