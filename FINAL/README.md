# FINAL...

##### importer .py
Converts .csv file into array using numpy. Will skip first row as it contains label description.

##### abstractMLAlgo .py
Contains abstract class for all ML implementations. Uses AbstractBaseClass for abstract implementation. 
  - train (train_data) : Classifier model trained
  - test (test_data) : Trained model tested with data
  - predict (predict_data) : Predicts labels for data
  - cross_validate (train_data) : Train and 10-fold classifier
 
##### scikit_NaiveBayes .py / scikit_NN .py
Sample implementation of GaussianNaiveBayes and NeuralNetworks using scikit-learn. These concrete classes should implement train(), test() and such of abstract base class.

#### main .py

Entry point of program. 
  - use importer .py to setup arrays of train_data and test_data 
  - loop through all the concrete classes to train(), test()...