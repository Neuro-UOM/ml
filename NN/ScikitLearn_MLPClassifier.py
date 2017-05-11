from sklearn.neural_network import MLPClassifier 
import numpy as np

data = np.loadtxt('train.csv', delimiter=',', dtype=np.float32)
X_ = data[:,:-1]
Y_ = data[:,-1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=2) # state = 2 ?
clf.fit(X_, Y_)

# print(Y_[3])
# print(clf.predict(X_[3:4]))
# print(clf.predict_proba(X_[3:4]) )

#######################################################
test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32)
test_X = test_data[:,:-1]
test_Y = test_data[:,-1]

#_____________________________________________________
correct = 0
incorrect = 0

for item in test_data:
    if(  clf.predict(item[:-1].reshape(1,-1)) == item[-1] ): 
        correct += 1
    else:
        incorrect += 1

print('Correct ', correct)
print('Incorrect ', incorrect)
print('\n')

#____________________________________________________
# left_test_data
print("Actual: ", test_Y[100], "Predicted: ", clf.predict(test_X[100:101])) # [100] in form of [[]] ???
print("Probabilities: ", clf.predict_proba(test_X[100:101]))

print("Actual: ", test_Y[1000], "Predicted: ", clf.predict(test_X[1000:1001])) 
print("Probabilities: ", clf.predict_proba(test_X[1000:1001]))