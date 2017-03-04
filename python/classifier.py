from abc import ABC, abstractmethod



class Classifier(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, example):
        pass
    
    @abstractmethod
    def evaluate(self, data):
        pass
