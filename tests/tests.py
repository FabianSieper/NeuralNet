
from layer.catCrossEntropy_test import catCrossEntropyTest
from layer.dense_test import denseTesting
from layer.meanSquaredError_test import squredTesting
from optimizer.adam_test import AdamUnitTests
from neuralNet_test import testNeuralNet, testShuffleData
from helper_test import testGetF1Score, testRecall, testPrecision    

def testAll():
    catCrossEntropyTest()
    denseTesting()
    AdamUnitTests()
    testNeuralNet()
    testRecall()
    testPrecision()
    testGetF1Score()
    testShuffleData()
    squredTesting()
    print("All tests ended successfully!")

if __name__=="__main__":

    testAll()