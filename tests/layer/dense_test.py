# ---------------------------------------------------
# UNIT TESTS
import sys
sys.path.append("../..")

from neuralNet import *
import numpy as np


def denseTesting():
    np.random.seed(2)


    # testing random initlization of layer
    denseLayer = Dense(None, 2, 3)
    expectedWeightShape = (2, 3)
    weightShape = denseLayer.getWeights().shape
    assert expectedWeightShape == weightShape

    # testing forward pass with random initialization
    x = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        ])
    denseLayer = Dense(None, 2, 1)
    weights = denseLayer.getWeights()
    bias = denseLayer.getBias()
    output = denseLayer.forward(x)
    expectedOutput = np.array([[0.7959693693799057], [1.150275927046334], [1.4171032021492005], [0.5291420942770391]])
    assert np.array_equal(output, expectedOutput)

    # testing dense layer with direct initalization of the dense layer
    weights = np.array([0.,1.])
    biases = np.array([1.])
    denseLayer = Dense(None, weights, biases)

    # testing the forward pass
    expectedOutput = np.array([[1],[2],[2],[1]])
    output = denseLayer.forward(x)
    assert np.array_equal(expectedOutput, output)

    # testing backward pass
    previousDerivatives = np.array([[5,6,7,8]]).T
    expectedDerivative = np.array([[0, 0, 0, 0],[5,6,7,8]]).T
    derivative = denseLayer.backward(previousDerivatives)
    assert np.all(expectedDerivative == derivative)

    # testing weight updates
    expectedWeightUpdates = np.array([[3,3.25]]).T
    weightUpdates = denseLayer.getWeightUpdates()
    assert np.all(expectedWeightUpdates == weightUpdates)

    # testing bias updates
    expectedBiasUpdates = np.array([6.5])
    biasUpdates = denseLayer.getBiasUpdates()
    assert np.all(biasUpdates == expectedBiasUpdates)

    lr = 0.5
    # test weight updates
    expectedNewWeights = np.array([[-1.5, -0.625]]).T
    denseLayer.updateWeights(lr)
    newWeights = denseLayer.getWeights()
    assert np.all(newWeights == expectedNewWeights)

    # test bias updates
    expectedNewBias = np.array([-2.25])
    denseLayer.updateBias(lr)
    newBias = denseLayer.getBias()
    assert np.all(newBias == expectedNewBias)

# for testing
if __name__ == "__main__":
    denseTesting()
