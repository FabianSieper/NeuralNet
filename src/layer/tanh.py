from .layer import Layer
import numpy as np

class Tanh(Layer):

    _outputValues = None

    def __init__(self, optimizer=None, *args):
        pass

    def forward(self, val):
        self._outputValues = (np.exp(val) - np.exp(-val)) / (np.exp(val) + np.exp(-val))
        return self._outputValues
    
    def backward(self, derivatives):
        self._gradient = (1 - np.power(self._outputValues, 2)) * derivatives
        return self._gradient

    def getGradient(self):
        return self._gradient
