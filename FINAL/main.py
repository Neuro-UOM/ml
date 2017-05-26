from importer import csvImporter
from scikit_NN import scikit_NN

dataImporter = csvImporter('train.csv')
train_data = dataImporter.getData()         # print(train_data[0])

test_data = csvImporter('test.csv').getData()

c = scikit_NN()
print(c.train(train_data))
print(c.test(test_data))
print(c.predict(train_data[100:101,:-1]))

input("Press Enter to continue...")