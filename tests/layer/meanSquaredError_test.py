# ---------------------------------------------------
# UNIT TESTS
import sys
sys.path.append("../..")

from neuralNet import *
import numpy as np

def squredTesting():

    sqError = MeanSquaredError()

    input_value = np.array([0,2,3,4])
    ground_truth = np.array([0,1,2,1])

    expectedMSE = 2.75 / 2
    MSE = sqError.forward(input_value, ground_truth)
    assert MSE == expectedMSE

    backwardOutput = sqError.backward(ground_truth)
    expectedBackwardOutput = np.array([0, -1, -1, -3])
    assert np.all(backwardOutput == expectedBackwardOutput)



    # testing on a whole neural net
    # testing basic init of neuralNet
    denseLayer = Dense(None, 3, 2)
    sigmoid = Sigmoid()
    lossLayer = MeanSquaredError()
    neuralNet = NeuralNet([denseLayer, sigmoid], lossLayer)

    inputValues = np.array([0,1,0.5])
    groundTruth = np.array([1,0])
    loss = neuralNet.forward(inputValues, groundTruth)


    loss = neuralNet.train(inputValues, groundTruth, lr = 0.2, shuffleData=False, iterations=100)


    prediction = neuralNet.forward(inputValues)

if __name__=="__main__":
    squredTesting()