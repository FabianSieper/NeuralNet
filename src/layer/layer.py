from abc import ABC

import numpy as np
class Layer(ABC):

    def __init__(self, optimizer, *args):
        self._outputValues = None

    def forward(self, val):
        return val

    def backward(self, derivatives):
        return None
    
    def updateWeights(self, lr):
        pass
    
    def updateBias(self, lr):
        pass

    def getOutputValues(self):
        if np.all(self._outputValues != None):
            return self._outputValues
        return None