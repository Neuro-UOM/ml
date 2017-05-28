from abstract_MLAlgo import MLAlgo
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

class scikit_KNearestNeighbor(MLAlgo):

    def __init__(self):
        self.clf = KNeighborsClassifier(n_neighbors=10)
        self.className = self.__class__.__name__ 

    def train(self, train_data):
        train_X = train_data[:,:-1]
        train_Y = train_data[:,-1]
        self.clf.fit(train_X, train_Y)
        print("KNeighborsClassifier model built.")
        return self.className + " Training finished...\n"
    
    def test(self, test_data):
        test_X = test_data[:,:-1]
        test_Y = test_data[:,-1]
        print("Accuracy: ", self.clf.score(test_X, test_Y))
        return self.className + " Testing finished...\n"

    def predict(self, predict_data):
        print("Predictions: ", self.clf.predict(predict_data))
        return self.className + " Prediction finished...\n"

    def cross_validate(self, train_data):
        X_ = train_data[:,:-1]
        Y_ = train_data[:,-1]
        predicted = cross_val_predict(self.clf, X_, Y_, cv=10)
        print("Cross-validation accuracy: ", metrics.accuracy_score(Y_, predicted))
        
        if metrics.accuracy_score(Y_, predicted) > MLAlgo.cross_validate_accuracy:
            MLAlgo.cross_validate_accuracy = metrics.accuracy_score(Y_, predicted)
            MLAlgo.classifier = self.clf
            
        return self.className + " Cross validation finished...\n"