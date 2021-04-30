from .layer import Layer
import numpy as np
class Sigmoid(Layer):

    _outputValues = None

    def __init__(self, optimizer=None, *args):
        pass

    def forward(self, val):

        np.seterr(all="raise")

        try:
            self._outputValues = 1. / (1. + np.e ** (-val))
        except Exception as e:
            # values above 985 raise an underflow in sigmoid
            # in this case
            np.seterr(all="ignore")
            self._outputValues = 1. / (1. + np.e ** (-val))
            self._outputValues[self._outputValues == 0] = 1

        np.seterr(all='warn')

        return self._outputValues
    
    def backward(self, derivatives):
        self._gradient = derivatives * self._outputValues * (1 - self._outputValues)
        return self._gradient

    def getGradient(self):
        return self._gradient

