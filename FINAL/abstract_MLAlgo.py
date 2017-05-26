from abc import ABCMeta, abstractmethod

class MLAlgo:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, train_data): pass

    @abstractmethod
    def test(self, test_data): pass

    @abstractmethod
    def predict(self, predict_data): pass