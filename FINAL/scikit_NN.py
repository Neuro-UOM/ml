from abstract_MLAlgo import MLAlgo
from sklearn.neural_network import MLPClassifier 

class scikit_NN(MLAlgo):

    def __init__(self):
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=2)
        self.className = self.__class__.__name__ 

    def train(self, train_data):
        train_X = train_data[:,:-1]
        train_Y = train_data[:,-1]
        self.clf.fit(train_X, train_Y)
        print("MLPClassifier model built.")
        return self.className + " Training finished...\n"
    
    def test(self, test_data):
        test_X = test_data[:,:-1]
        test_Y = test_data[:,-1]
        print("Accuracy: ", self.clf.score(test_X, test_Y))
        return self.className + " Testing finished...\n"

    def predict(self, predict_data):
        print("Predictions: ", self.clf.predict(predict_data))
        return self.className + " Prediction finished...\n"