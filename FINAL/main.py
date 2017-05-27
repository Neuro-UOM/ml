from importer import csvImporter
from abstract_MLAlgo import MLAlgo
from scikit_NeuralNetwork import scikit_NeuralNetwork
from scikit_NaiveBayes import scikit_NaiveBayes
from scikit_DecisionTree import scikit_DecisionTree
from scikit_SupportVectorMachine import scikit_SupportVectorMachine
#from scikit_GaussianProcessClassifier import scikit_GaussianProcessClassifier

train_data = csvImporter('train.csv').getData()    
test_data = csvImporter('test.csv').getData()

for ML_Class in MLAlgo.__subclasses__():
    print("*** ", ML_Class.__name__, " ***\n")

    MLmodel = ML_Class()
    print(MLmodel.train(train_data))
    print(MLmodel.test(test_data))
    print(MLmodel.predict(train_data[100:101,:-1]))
    print(MLmodel.cross_validate(train_data))

input("Press Enter to continue...")

#c = scikit_NN()
#print(c.train(train_data))
#print(c.test(test_data))
#print(c.predict(train_data[100:101,:-1]))
#print(c.cross_validate(train_data))

#d = scikit_NaiveBayes()
#print(d.train(train_data))
#print(d.test(test_data))
#print(d.predict(train_data[100:101,:-1]))
#print(d.cross_validate(train_data))