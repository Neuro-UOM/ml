from abc import ABCMeta, abstractmethod, abstractproperty

class MLAlgo:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, train_data): pass

    @abstractmethod
    def test(self, test_data): pass

    @abstractmethod
    def predict(self, predict_data): pass

    @abstractmethod
    def cross_validate(self, train_data): pass
    
    cross_validate_accuracy = 0
    classifier = object
    trained_instance = object

    '''
    @abstractproperty
    def cross_validate_accuracy(self): pass

    @abstractproperty
    def classifier(self): pass
    '''