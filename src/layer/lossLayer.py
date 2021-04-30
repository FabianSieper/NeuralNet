from abc import ABC

class LossLayer(ABC):

    def forward(self, y, predictions):
        pass

    def backward(self, y):
        pass

