# ----------------------------
# UNIT TESTS
import sys
sys.path.append("../..")
import numpy as np
from neuralNet import *

def catCrossEntropyTest():
    
    predictions = np.array([[0.2, 0.4, 0.5, 0.6], [0.1, 0.1, 0.2, 1.2]])
    # softmax values would be: 
    # 0.1975079917625, 0.2412368058974, 0.26660790224726494, 0.29464730009283224
    # 0.16368388410887322, 0.16368388410887322, 0.1808986684747913, 0.4917335633074623

    y = np.array([[0, 0, 0, 1], [0, 1, 0, 0]])
    crossEntropy = CategoricalCrossEntropy()

    # testing forward pass
    expectedCrossEntropy = 1.5158972390034036
    crossEntropyValue = crossEntropy.forward(predictions, y)
    assert np.array_equal(expectedCrossEntropy, crossEntropyValue)

    # testing backward pass
    expectedBackwardValue = np.array([[0.1975079917625, 0.2412368058974, 0.26660790224726494, -0.70535269990716776],
                                    [0.16368388410887322, -0.83631611589112678, 0.1808986684747913, 0.4917335633074623]])
    backwardValue = crossEntropy.backward(y)
    assert np.all(np.round_(expectedBackwardValue, 6) == np.round_(backwardValue,6))


if __name__ == "__main__":
    catCrossEntropyTest()