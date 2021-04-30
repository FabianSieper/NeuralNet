# ---------------------------------------------
# UNIT TESTS

# sys.path.append is required in order to make use of a package which is lying beneath this package (example)
# sys.path.append is required in order to make use of a package which is lying beneath this package (example)
import sys
sys.path.append("../../../..")
from neuralNet import *
import numpy as np

def testShuffleData():

    values_old = np.array([[1, 1], [2, 2], [3, 3],[4, 4]])
    values1 = np.array([[1, 1], [2, 2], [3, 3],[4, 4]])
    values2 = np.array([[1, 1], [2, 2], [3, 3],[4, 4]])

    NeuralNet.shuffleData(values1, values2)
    assert np.all(values1 == values2)
    assert np.any(values_old != values1)
    
    x = np.array([[1, 1], [2, 2], [3, 3],[4, 4]])
    y = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4]])

    NeuralNet.shuffleData(x, y)

    assert np.all(x != None)
    assert np.all(y != None)
    assert np.all([x[i][0] == y[i][0] for i in range(len(x))])

    values = np.array([[1, 2], [3, 4]])
    NeuralNet.shuffleData(values, values)

    assert np.all(values[0] == np.array([1,2])) or np.all(values[1] == np.array([1,2]))
def testNeuralNet():
    np.random.seed(2)

    # testing basic init of neuralNet
    denseLayer = Dense(None, 2,1)
    sigmoid = Sigmoid()
    lossLayer = CategoricalCrossEntropy()
    neuralNet = NeuralNet([denseLayer, sigmoid], lossLayer)

    assert neuralNet.getLayer() == [denseLayer, sigmoid]
    assert neuralNet.getLossLayer() == lossLayer

    # forward pass test Nr. 1
    x = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        ])
    y = np.array([[0], [0], [1], [1]])
    denseLayer = Dense(None, 2,1)
    sigmoid = Sigmoid()
    lossLayer = CategoricalCrossEntropy()
    neuralNet = NeuralNet([denseLayer, sigmoid], lossLayer)

    # dense Layer output : 0.765657213622151 0.75070262309136313 1.18602501570964003 0.3303348210038741
    # sigmoid output : 0.6825807139947168 0.679331777951338291 0.7660293878022749970 0.58184084171891924
    # softmax output : 1 1 1 1
    # categorical cross entropy values : 0 0 0 0 

    loss, softmaxValues = neuralNet.forward(x, y)
    expectedLoss = np.array([0])
    expectedSoftmaxValues = np.array([[1], [1], [1], [1]])
    assert np.all(expectedLoss == loss)
    assert np.all(softmaxValues == expectedSoftmaxValues)

    # forward pass test Nr. 2
    x = np.array([
        [1, 0, 0.5],
        [0, 1, 0.5]
        ])
    y = np.array([[0,1], [1,0]])
    denseLayer = Dense(None, 3, 2)
    sigmoid = Sigmoid()

    # dense layer output: (0.64979549576742351 1.39742013475492965) (0.74480153540410415 1.04497644350713258)
    # sigmoid output: (0.656964376632429293 0.8017741837305471702) (0.678044926260500659 0.7398090722680921276)
    # softmax output: (0.463860679266887663 0.536139320733112337) (0.484563870344268877 0.515436129655731123)
    # categorical cross entropy values : (0, 0.623361224945556576) (0.724506029003823532 0)
    lossLayer = CategoricalCrossEntropy()
    neuralNet = NeuralNet([denseLayer, sigmoid], lossLayer)

    loss, softmaxValues = neuralNet.forward(x, y)
    expectedLoss = 0.673933626974690054
    expectedSoftmaxValues = np.array([[0.463860679266887663, 0.536139320733112337], 
                                    [0.484563870344268877, 0.515436129655731123]])

    assert np.round(loss, 8) == np.round(expectedLoss, 8)
    assert np.all(np.round(expectedSoftmaxValues, 8) == np.round(softmaxValues, 8))

    # backward pass test

    # dense layer output: (0.64979549576742351 1.39742013475492965) (0.74480153540410415 1.04497644350713258)
    # sigmoid output: (0.656964376632429293 0.8017741837305471702) (0.678044926260500659 0.7398090722680921276)
    # softmax output: (0.463860679266887663 0.536139320733112337) (0.484563870344268877 0.515436129655731123)
    # categorical cross entropy values : (0 0.623361224945556576) (0.724506029003823532 0)
    lossLayer = CategoricalCrossEntropy()
    neuralNet = NeuralNet([denseLayer, sigmoid], lossLayer)

    loss, softmaxValues = neuralNet.forward(x, y)
    neuralNet.backward()

    # gradient of cat. cross entropy (and softmax): (0.463860679266887663  -0.463860679266887663) 
    #                                               (-0.515436129655731123  0.515436129655731123)
    # gradient of sigmoid: (0.104536655968578364 -0.0737224641332581448) 
    #                      (-0.11251970928563202891 0.0992171298609295714560)
    # input: (1 0 0.5)
    #        (0 1 0.5)
    # bias updates:   (-0.003991526658526832455 0.01274733286383571332800)
    # weight updates: (0.104536655968578364 -0.0737224641332581448 0.0154070959176601096)
    #                 (-0.11251970928563202891 0.0992171298609295714560 -0.006651289712351228727)
    biasUpdate = denseLayer.getBiasUpdates()
    expectedBiasUpdates = np.array([-0.003991526658526832455, 0.01274733286383571332800])
    
    weightUpdate = denseLayer.getWeightUpdates()
    expectedWeightUpdates = np.array([[0.052268327984289182, -0.0368612320666290724],
                                        [-0.056259854642816014455, 0.049608564930464785728],
                                        [-0.0019957633292634162275000, 0.006373666431917856664]])
    assert np.all(np.round(biasUpdate, 8) == np.round(expectedBiasUpdates, 8))
    assert np.all(np.round(weightUpdate, 7) == np.round(expectedWeightUpdates, 7))


    # test update
    neuralNet.update(0.1)
    expectedNewWeights = np.array([[0.1994217972015710818, 0.62295709320666290724],
                                    [0.30528065546428160144550, 0.2618664235069535214272],
                                    [0.62133340633292634162275, 0.5285047233568082143336]])
    expectedNewBias = np.array([0.1349791026658526832455, 0.5123033867136164286672])
    assert np.all(np.round(expectedNewWeights, 7) == np.round(denseLayer.getWeights(), 7))
    assert np.all(np.round(expectedNewBias, 7) == np.round(denseLayer.getBias(), 7))

if __name__=="__main__":
    testNeuralNet()
    testShuffleData()