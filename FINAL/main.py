from importer import csvImporter
from abstract_MLAlgo import MLAlgo
from scikit_NN import scikit_NN
from scikit_NaiveBayes import scikit_NaiveBayes

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