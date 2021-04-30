# --------------------------------------
# UNIT TESTS
import sys
sys.path.append("../..")

from neuralNet import *
import numpy as np


def AdamUnitTests():

    weights = np.array([[0.5, 0.5, 0.5],[0.2, 0.1, 0.3]])
    gradient = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.2]])
    lr = 0.5
    beta_1 = 0.5
    beta_2 = 0.1
    epsilon = 0.1
    adam = Adam(beta_1, beta_2, epsilon)

    # testing function "optimize"
    # m = {{0.05, 0.05, 0.05}, {0.1, 0.05, 0.1}}
    # v = {{0.009, 0.009, 0.009}, {0.036, 0.009, 0.036}}
    # m_hat = {{0.1, 0.1, 0.1}, {0.2, 0.1, 0.2}}
    # v_hat = {{0.01, 0.01, 0.01}, {0.04, 0.01, 0.04}}
    # output = {{0.25, 0.25, 0.25}, {-0.133333, -0.15, -0.0333333}}

    updatedWeights = adam.optimize(weights, gradient, lr)
    expectedUpdatedWeights = np.array([[0.25, 0.25, 0.25], [-0.133333, -0.15, -0.0333333]])

    assert np.all(np.round(updatedWeights, 6) == np.round(expectedUpdatedWeights, 6))

    # testing second iteration of function "optimize"
    # m = {{0.075, 0.075, 0.075}, {0.15, 0.075, 0.15}}
    # v = {{0.0099, 0.0099, 0.0099}, {0.0396, 0.0099, 0.0396}}
    # m_hat = {{0.1, 0.1, 0.1}, {0.2, 0.1, 0.2}}
    # v_hat = {{0.01, 0.01, 0.01}, {0.04, 0.01, 0.04}}
    # output = {{0.25, 0.25, 0.25}, {-0.133333, -0.15, -0.0333333}}
    updatedWeights = adam.optimize(weights, gradient, lr)
    expectedUpdatedWeights = np.array([[0.25, 0.25, 0.25], [-0.133333, -0.15, -0.0333333]])

    assert np.all(np.round(updatedWeights, 6) == np.round(expectedUpdatedWeights, 6))

    # testing third iteration of function "optimize"
    gradient = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.2]])
    # m = {{0.2875, 0.2875, 0.2875}, {0.125, 0.0875, 0.175}}
    # v = {{0.22599, 0.22599, 0.22599}, {0.01296, 0.00999, 0.03996}}
    # m_hat = {{0.328571, 0.328571, 0.328571}, {0.142857, 0.1, 0.2}}
    # v_hat = {{0.226216, 0.226216, 0.226216}, {0.012973, 0.01, 0.04}}
    # output = {{0.214595, 0.214595, 0.214595}, {-0.133936, -0.15, -0.0333333}}
    updatedWeights = adam.optimize(weights, gradient, lr)
    expectedUpdatedWeights = np.array([[0.214594, 0.214594, 0.214594], [-0.133936, -0.15, -0.0333333]])

    assert np.all(np.round(updatedWeights, 6) == np.round(expectedUpdatedWeights, 6))


if __name__ == "__main__":
    AdamUnitTests()