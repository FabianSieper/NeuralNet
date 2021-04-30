

from .layer import Layer
import numpy as np


class Dense(Layer):

    _inputValues = None
    _weightUpdates = None
    _biasUpdates = None

    def __init__(self, optimizer=None, *args):

        """ takes:
                - optimizer: Optimizer object
                    - if object == None: a default optimization strategy is used
                    - a copy of the given obect will be stored for both, weights and biases
                - args: either two np.ndarrays with weights, or two integers which describe the amount of ingoing
                        and outgoing connections of the layer. In this case the weights are set randomly """

        if len(args) != 2:
            print("The intialization of the dense layer requires three arguments!")
            exit()

        self._weights = None
        self._bias = None

        # if an optimizer was given (not None) set the correspoding values of this object
        if optimizer:
            self._weightOptimizer = optimizer.copy()
            self._biasOptimizer = optimizer.copy()
        else:
            self._weightOptimizer = None
            self._biasOptimizer = None
            
        # if weights and biases are given directly
        if type(args[0]) == np.ndarray:
            self._weights = args[0][:,np.newaxis]
            self._bias = args[1]
        # if the shape is given the weights have to bi initialized randomly
        elif type(args[0]) == int and type(args[1]) == int:
            self._weights = np.random.random(args[0] * args[1]).reshape(args[0], args[1])
            self._bias = np.random.random(args[1]).reshape(1,args[1])

        else:
            print("ERROR - the given args are not able to initialize the Dense layer: " + str(args))
            exit()

    def forward(self, val):
        """ takes: 
                - val: np.array of input values for the forward pass
            does: 
                - computes the output of the layer, and stores it in its own object additionally
            returns:
                - the computed value from the forward pass """

        np.seterr(all="ignore")
        self._inputValues = val
        self._outputValues = self._inputValues @ self._weights + self._bias
        np.seterr(all="warn")
        return self._outputValues

    def backward(self, derivatives):

        if np.all(self._inputValues == None) or np.all(self._bias == None):
            raise Exception("Cant update weights of dense layer, as no forwardpass was called yet ... ")


        self._weightUpdates = (self._inputValues.T @ derivatives) / self._inputValues.shape[0]

        self._biasUpdates = np.mean(derivatives, axis=0)
        self.derivative = derivatives @ self._weights.T

        return self.derivative

    def updateBias(self, lr):
        # if no optimizer was set - use the default optimization strategy
        if not self._biasOptimizer:
            self._bias -= lr * self._biasUpdates
        else:
            self._bias = self._biasOptimizer.optimize(self._bias, self._biasUpdates, lr)
            
    def updateWeights(self, lr):
        # if no optimizer was set - use the default optimization strategy
        if not self._weightOptimizer:
            self._weights -= lr * self._weightUpdates
        else:
            self._weights = self._weightOptimizer.optimize(self._weights, self._weightUpdates, lr)

    def setWeights(self, weights):
        self._weights = weights

    def setBias(self, bias):
        self._bias = bias

    def getWeights(self):
        return self._weights

    def getBias(self):
        return self._bias

    def getWeightUpdates(self):
        return self._weightUpdates

    def getBiasUpdates(self):
        return self._biasUpdates

    def getOutputValues(self):
        return self._outputValues

