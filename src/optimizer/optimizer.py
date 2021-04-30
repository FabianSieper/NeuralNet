
from abc import ABC

class Optimizer(ABC):

    def optimize(self, weights, gradient, lr):
        """ takes:
                - weights: weights of the neural net which are to be updated -> np.ndarray
                - gradient: the derivative of the loss function after the weights -> np.ndarray
                - lr: learning rate -> float
            does:
                - internally computes the updated weights
            returns:
                - updated weights """
        pass