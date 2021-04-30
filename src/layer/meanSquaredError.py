from .lossLayer import LossLayer
import numpy as np

class MeanSquaredError(LossLayer):

    def __init__(self):
        self._prevPredictions = None
        self._prevGroundTruth = None
        self._gradient = None

    def forward(self, predictions, y=None):
        """ takes:
                - predictions: values which are to be put the mse -> np.ndarray
                    - if y == None, 'predictions' are returned
                - y: ground_truth -> np.ndarray
            returns:
                - the mean squared error -> float """

        if np.all(y == None):
            return predictions

        self._prevPredictions = predictions
        self._prevGroundTruth = y

        self._meanSqError = np.sum((y - predictions) ** 2) / len(predictions) / 2
        return self._meanSqError

    def backward(self, y):

        if np.all(self._prevPredictions == None) or np.all(self._prevGroundTruth == None):
            raise ValueError("No forward pass was done before the backward pass was initiated!")

        self._gradient = y - self._prevPredictions
        return self._gradient

    def getGradient(self):

        if np.all(self._gradient == None):
            raise ValueError("The backward pass has to be called before a gradient cand be returned!")

        return self._gradient

