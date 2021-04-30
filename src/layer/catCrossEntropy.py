from .lossLayer import LossLayer
import numpy as np

def softmax(o):
    o_new = o
    maxVal = np.max(o)
    sum_val = np.sum(np.e ** (o_new - maxVal), axis=1, keepdims=True)
    return ((np.e ** (o_new - maxVal)) / sum_val)

def categorical_cross_entropy(h, y_one_hot):

    h = np.clip(h, a_min=0.000000001, a_max=None)
    catCrossEntropy =  -1 * np.sum(y_one_hot * np.log(h), axis=1)
    averageCrossEntropy = np.sum(catCrossEntropy, axis=0) / len(h)
    return averageCrossEntropy

class CategoricalCrossEntropy(LossLayer):
    """ this layer automatically contains a softmax functionality """

    def __init__(self):
        pass

    def forward(self, predictions, y = None):
        """ takes:
                - predictions: values which are to be put into the softmax layer -> np.ndarray
                - y: ground_truth -> np.ndarray
                    - can be None, when a prediction is wanted, and no ground truth is known
            returns:
                - cross entropy (if y is != None) -> np.ndarray
                - softmax (if y == None) -> np.ndarray """

        self._soft = softmax(predictions)
        
        if np.all(y != None):
            self._crossEntropy = categorical_cross_entropy(self._soft, y)
            return self._crossEntropy

        else:
            return self._soft
    
    def backward(self, y):
        self._gradient = self._soft - y
        return self._gradient

    def getSoftmaxValues(self):
        return self._soft

    def getGradient(self):
        return self._gradient

