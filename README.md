# ccgnet
CCGNet (Co-Crystal Graph Network) is a deep learning framework for virtual screening of organic cocrystals, which integrates the priori knowledge into the feature learning on molecular graph and achieves a great improvement of performance for data-driven cocrystal screening.

Requirements:
* Python 3.7
* Tensorflow (>=1.6)
* RDKit
* Openbabel 2.4.1
* CCDC Python API
* Scikit-learn

The ccgnet module could be used to build CCGNet models. In ccgnet, the message passing is implement by Graph-CNN, where the code of this part we borrow from https://github.com/fps7806/Graph-CNN. The Featurize module represents the combination of two molecular as the input of CCGNet models.
