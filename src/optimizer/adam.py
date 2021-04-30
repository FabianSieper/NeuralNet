
from .optimizer import Optimizer
import numpy as np

class Adam(Optimizer):

    def __init__(self, beta_1, beta_2, epsilon, lambdaReg = 0):
        """ takes:
                - beta_1: scalar for scaling the weight of the momentum
                - beta_2: scalar for scaling the weight of the acceleration 
                - epsilon: bias for division (makes sure there is not division by zero) 
                - lambdaReg: weight for l2-regularization. It set to 0, no regularization will be done """


        self.beta_1 = beta_1
        self.beta_2 = beta_2 
        self.epsilon = epsilon
        self.lambdaReg = lambdaReg 

        self._m = 0
        self._v = 0
        self._iterations = 1

    def optimize(self, weightsOrBias, gradient, lr):
        """ takes:
                - weightsOrBias: weights or biases of the neural net which are to be updated -> np.ndarray
                - gradient: the derivative of the loss function after the weights -> np.ndarray
                - lr: learning rate -> float
            does:
                - internally computes the updated weights
            returns:
                - updated weights """


        self._m = self.beta_1 * self._m + (1 - self.beta_1) * gradient
        self._v = self.beta_2 * self._v + (1 - self.beta_2) * gradient ** 2

        m_hat = self._m / (1 - self.beta_1 ** self._iterations)
        v_hat = self._v / (1 - self.beta_2 ** self._iterations)

        # update the amount of update steps there were
        self._iterations += 1

        return weightsOrBias - lr * m_hat / (np.sqrt(v_hat) + self.epsilon) - self.lambdaReg * weightsOrBias

    def copy(self):

        copy = Adam(self.beta_1, self.beta_2, self.epsilon)
        copy.setM(self._m)
        copy.setV(self._v)
        copy.setIterations(self._iterations)
        copy.setLambdaReg(self.lambdaReg)
        return copy

    def setM(self, m):        
        self._m = m
    
    def setV(self, v):
        self._v = v

    def setLambdaReg(self, lambdaReg):
        self.lambdaReg = lambdaReg

    def setIterations(self, iterations):
        self._iterations = iterations