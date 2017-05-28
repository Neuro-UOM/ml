from importer import csvImporter
from abstract_MLAlgo import MLAlgo

from scikit_NeuralNetwork import scikit_NeuralNetwork
from scikit_NaiveBayes import scikit_NaiveBayes
from scikit_DecisionTree import scikit_DecisionTree
from scikit_SupportVectorMachine import scikit_SupportVectorMachine
from scikit_KNearestNeighbor import scikit_KNearestNeighbor
from scikit_NearestCentroid import scikit_NearestCentroid
from scikit_StochasticGradientDescent import scikit_StochasticGradientDescent
from scikit_GaussianProcessClassifier import scikit_GaussianProcessClassifier

train_data = csvImporter('train.csv').getData()    
test_data = csvImporter('test.csv').getData()

#iris_data = csvImporter('IRIS.csv').getData()
#iris_train_data = iris_data[0:80]
#iris_test_data = iris_data[80:100]
#iris_predict_data = iris_data[80:81,:-1]

for ML_Class in MLAlgo.__subclasses__():
    print("*** ", ML_Class.__name__, " ***\n")

    MLmodel = ML_Class()
    print(MLmodel.train(train_data))
    print(MLmodel.test(test_data))
    print(MLmodel.predict(train_data[100:101,:-1]))
    print(MLmodel.cross_validate(train_data))

input("Press Enter to continue...")


'''
http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

iris_X = digits.data
iris_Y = digits.target
'''

#c = scikit_NN()
#print(c.train(train_data))
#print(c.test(test_data))

#d = scikit_NaiveBayes()
#print(d.train(train_data))
#print(d.test(test_data))