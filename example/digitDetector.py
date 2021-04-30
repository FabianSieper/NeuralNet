import mnist_downloader
from mnist import MNIST
import numpy as np
import os

# sys.path.append is required in order to make use of a package which is lying beneath this package (example)
import sys
sys.path.append("..")


from src.scaler import StandardScaler, NormalScaler
from src.layer.dense import Dense
from src.layer.sigmoid import Sigmoid
from src.layer.catCrossEntropy import CategoricalCrossEntropy
from src.neuralNet import NeuralNet
from src.optimizer.adam import Adam
from src.helper import createOnHotVectors, getAccuracy, getF1Score, getRecall, getPrecision, computeAllPerformanceStats
from src.helper import visualizeHistory

# import all parts to build a neural net
# layers (Dense, Sigmoid, CategoricalCrossEntropy, ...)
# the neuralNet itself (NeuralNet)
# optimizer (Adam)
# from ..src import

def getMnistData():

    download_folder = "./mnist/"
    mnist_downloader.download_and_unzip(download_folder)

    mndata = MNIST('mnist', return_type="numpy")

    images_train, labels_train = mndata.load_training()
    images_validation, labels_validation = mndata.load_testing()

    return images_train, labels_train, images_validation, labels_validation


def train_digit_detector(images_train, labels_train, images_validation, labels_validation):

    x = np.array(images_train)
    y = createOnHotVectors(labels_train, 10)    

    adam = Adam(0.5, 0.5, 0.1)

    denseLayer1 = Dense(adam, 784, 12)
    sigmoid1 = Sigmoid()
    denseLayer2 = Dense(adam, 12, 10)
    lossLayer = CategoricalCrossEntropy()
    layer = [denseLayer1, sigmoid1, denseLayer2]
    neuralNet = NeuralNet(layer, lossLayer)

    # scale input values by using the built-in-method of the neuralNet
    neuralNet.setScaler(StandardScaler())
    neuralNet.fitScaler(x)
    
    x = neuralNet.scaleValues(x)

    lr = 0.8
    iterations = 4
    batch_size = 1000
    amountTestData = 0.02

    performanceValues = neuralNet.train(x, 
                                         y, 
                                         lr, 
                                         amountTestData = amountTestData,
                                         batch_size = batch_size, 
                                         iterations = iterations, 
                                         storePerformanceHistory = True,
                                         gatherPerformanceHistoryPerBatch = True,
                                         printPerformanceStats = False)


    visualizeHistory(performanceValues["loss"], label="Error")
    visualizeHistory(performanceValues["accuracy"], label="Accuracy")

    neuralNet.save("checkpoint.dump")
    neuralNet.load("checkpoint.dump")

    

