
from .layer.lossLayer import LossLayer
from .layer.catCrossEntropy import CategoricalCrossEntropy
from .helper import computeAllPerformanceStats

import numpy as np
import math
from tqdm import tqdm
import pickle

class NeuralNet:

    def __init__(self, layer: list, lossLayer: LossLayer):
        """ takes:
                - layer: list of Layers
                - lossLayer: single loss layer """
        self._layer = layer
        self._lossLayer = lossLayer
        self._scaler = None

    def train(self, 
                x, 
                y, 
                lr,
                amountTestData = 0.1, 
                batch_size = -1, 
                iterations = 1, 
                shuffleData = True,
                storePerformanceHistory = False,
                gatherPerformanceHistoryPerBatch = False,
                printPerformanceStats = False):
                 
        """ takes:
                - x: train data -> np.ndarray
                - y: ground Truth in form of a one-hot-vector -> np.ndarray
                - lr: learning rate -> float
                - amountTestData: tells which percentage of the data should be taken for perfromance testing -> float
                    - if parameter < 0, the train-data will be used to evaluate the model
                - batch_size: how many data samples shall be used at once to train the network
                - shuffleData: is the data supposed to be shuffled before training? -> bool
                - iterations: how often the neural net shall be trained on the data set
                - storePerformanceHistory: variable tells, if losses per iteration shall be stored and returned after training -> bool
                - gatherPerformanceHistoryPerBatch: tells the program if the performance of the NN shall be evaluated after each batch-training -> bool
                - printPerformanceStats: tells if information about the performance of the current iteration is to be printed to the cmd -> bool
            does: 
                - trains the neural net with the given data
            returns:
                - performanceValues: a dict containing multiple keys regarding the performance of the trained model:
                    - 'loss' -> loss values gathered per iteration or per batch (choosable by parameter 'gatherPerformanceHistoryPerBatch')
                    - 'accuracy' -> overall accuracy of the model
                    - 'f1' -> f1 score of the indiviual ground_truths
                    - 'recall' -> recall of the individual ground_truths
                    - 'precision' -> precision of the individual ground_truths """

        # if batch size is not set, or too big, only use one batch
        if batch_size < 0 or batch_size > x.shape[0]:
            batch_size = x.shape[0]

        losses = []
        accuracies = []
        f1_scores = []
        performanceValues = {"loss" : [], "accuracy" : [], "f1" : [], "recall" : [], "precision" : []}

        # shuffle data
        x_cpy = x.copy()
        y_cpy = y.copy()
        if shuffleData:
            NeuralNet.shuffleData(x_cpy, y_cpy)

        # compute fraction used for training
        # if amountTestData < 0, the train-data shall be used for evaluation
        # in this case the model can be trained on the whole data, thus the fraction is '0'
        trainDataFraction = 1
        if amountTestData > 0:
            trainDataFraction = (1 - amountTestData)

        for it in range(iterations):

            # for each batch of training data (excluded the amount of data required for testing)
            # makes use of the first trainDataFraction percent of the whole data ...
            for i in tqdm(range(0, math.floor(x_cpy.shape[0] * trainDataFraction) - batch_size + 1, batch_size), "Training on batches at iteration " + str(it + 1) + "/" + str(iterations) + " ... "):
                
                self.forward(x_cpy[i : i + batch_size], y_cpy[i : i + batch_size])
                self.backward()
                self.update(lr)

                # if information about performance of the nn shall be gathered per batch
                if gatherPerformanceHistoryPerBatch:
                    # gather data for performance anlysis based on the set percentage of test data
                    if storePerformanceHistory: 
                        loss, accuracy, f1, recall, precision = self._getPerformanceStatsOnTestData(amountTestData, x_cpy, y_cpy)
                        performanceValues["loss"].append(loss)
                        performanceValues["accuracy"].append(accuracy)
                        performanceValues["f1"].append(f1)
                        performanceValues["recall"].append(recall)
                        performanceValues["precision"].append(precision)
                        

            # if information about performance of the nn shall NOT be gathered per batch (but per iteration)
            # or if there is only one batch - informations about performance has to be gathered between iterations not between batches
            if (not gatherPerformanceHistoryPerBatch or batch_size == x_cpy.shape[0]) and storePerformanceHistory:
                loss, accuracy, f1, recall, precision = self._getPerformanceStatsOnTestData(amountTestData, x_cpy, y_cpy)
                performanceValues["loss"].append(loss)
                performanceValues["accuracy"].append(accuracy)
                performanceValues["f1"].append(f1)
                performanceValues["recall"].append(recall)
                performanceValues["precision"].append(precision)

            # if performance stats shall be printed on cmd
            if printPerformanceStats and storePerformanceHistory:
                if len(accuracies) > 0 and len(losses) > 0:
                    print("INFO - Accuracy: " + str(accuracies[-1]))
                    print("INFO - Loss: " + str(losses[-1]))


        return performanceValues


    def _getPerformanceStatsOnTestData(self, amountTestData, x, y):
        """ takes:
                - amountTestData: percentage how much of the trainings data shall be used -> float
                - x: whole data -> np.ndarray
                - y: whole ground truth for data -> np.ndarray
            does:
                - takes last "amountTestData" percentage of x and y to gain information about accuracy, loss and the f1 score
            returns:
                - loss -> float
                - accuracy -> float
                - f1-score -> np.ndarray
                - recall -> np.ndarray
                - precision -> np.ndarray """

        loss = None
        softmaxValues = None

        if amountTestData < 0:
            loss, softmaxValues = self.forward(x, y)
            accuracy, f1, recall, precision = computeAllPerformanceStats(softmaxValues, y)
            return loss, accuracy, f1, recall, precision

        else:
            loss, softmaxValues = self.forward(x[math.floor(x.shape[0] * (1 - amountTestData)) : x.shape[0]],
                                        y[math.floor(x.shape[0] * (1 - amountTestData)) : x.shape[0]])
            accuracy, f1, recall, precision = computeAllPerformanceStats(softmaxValues, y[math.floor(x.shape[0] * (1 - amountTestData)) : x.shape[0]])
            return loss, accuracy, f1, recall, precision


            
    def forward(self, x, y = None):
        """ takes:
                - x: np.ndarray of input values
                - y: np.ndarray of ground Truth
            does:
                - computes the categorical cross entropy and softmax (if y != None)
                - computes softmax (if y == NOne)
            returns: 
                - if y != None:
                    - loss
                    - softmax values
                - if y == None:
                    - softmax values """

        # simple predictions dont need a label
        if np.all(y != None):
            self._lastY = y

        temp = x
        for layer in self._layer:
            temp = layer.forward(temp)
        
        if np.all(y != None):
            # if ground truth is known
            loss = self._lossLayer.forward(temp, y)

            # as the softmax-layer in integrated into the CCE, in this case the softmax-values shall be returned
            if isinstance(self._lossLayer, CategoricalCrossEntropy):
                softmaxValues = self._lossLayer.getSoftmaxValues()
                return loss, softmaxValues
            else:
                return loss, self._layer[-1].getOutputValues()

        else:
            # if ground truth is not known, return the output of the loss-layer
            # for normal the output of the loss layer, when not y is given, is the input of the loss layer,
            # or in the case of the CCE, the value of the softmax
            return self._lossLayer.forward(temp)

    def backward(self, y = None):
        """ takes:
                - y: np.ndarry of labels
                    - if set to "None" the y used in the forwardpass will be used 
            does:
                - computes the derivative of all layers 
            returns:
                - nothing """
        # if no specific new labels where set between the call of 'forward' and 'backward', use the labels
        # used in function 'forward'
        if np.all(y != None) and type(y) == np.ndarray:
            self._lastY = y

        gradient = self._lossLayer.backward(self._lastY)

        # iterate from last layer to the first layer
        for i in range(len(self._layer) - 1, -1, -1):
            gradient = self._layer[i].backward(gradient)

    def shuffleData(x, y):
        """ does:
                - shuffles the given input data 'x' and the corresponding ground truth y equally
            takes:
                - x -> np.ndarray
                - y -> np.ndarray """

        if x.shape[0] != y.shape[0]:
            print("ERROR - The given data does not have the same shape and thus is not able to be shuffled equally!")
            exit()

        random_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(random_state)
        np.random.shuffle(y)


    def update(self, lr):
        """ takes:
                - lr: learning rate for updating bias and weight -> float
            does:
                - the weights and biases of all layers are being updated """

        for layer in self._layer:
            
            layer.updateWeights(lr)
            layer.updateBias(lr)

    def setScaler(self, scaler):
        """ takes:
                - scaler: Scaler-object
            does:
                - sets the scaler object for the neural net """

        self._scaler = scaler

    def fitScaler(self, x):
        """ takes:
                - x: values which are to be put into the neural net
            does: 
                - sets scaler object in NeuralNeut-object for future fitting """

        self._scaler.fit(x)

    def scaleValues(self, x):
        """ takes:
                - x: values which are to be put into the neural net
            does:
                - uses the previously set Scaler-object (set by method 'setScaler') to scale x values
            returns:
                - rescaled x-values """

        if self._scaler == None:
            print("WARNING - No scaler has been set yet. Use 'fitScaler' to set a scaler.\nReturning values without manipulation ...")
            return x

        return self._scaler.transform(x.copy())


    def load(self, path):
        """ takes:
                - path: path describes where the neuralNet shall be loaded from
            does:
                - restores the neuralNet-object from the given file (path)
            returns:
                - bool: TRUE if loading process was successful, FALSE if not """


        print("INFO - Loading neuralNet from '" + path + "' ...")

        try:
            with open(path, "rb") as output:
                loadedNet = pickle.load(output)
                
                # restoring this object
                self._layer = loadedNet.getLayer()
                self._lossLayer = loadedNet.getLossLayer()
                self._scaler = loadedNet.getScaler()

        except Exception as e:
            print("WARNING - Not able to load neuralNet ... ")
            print(e)

            return False
        
        print("INFO - Successfully loaded neuralNet ...")
        return True

    def save(self, path):
        """ takes:
                - path: path describes where the neuralNet shall be saved at 
            does:
                - saves the neuralNet to the given path 
            returns:
                - bool: TRUE if saving process was successful, FALSE if not """

        print("INFO - Saving neuralNet to '" + path + "' ...")

        try:
            with open(path, "wb") as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("WARNING - Not able to save neuralNet ... ")
            print(e)
            return False

        print("INFO - Successfully saved neuralNet ... ")
        return True

    def getLayer(self):
        return self._layer

    def getLossLayer(self):
        return self._lossLayer

    def getSoftmaxValues(self):
        return self._lossLayer.getSoftmaxValues()

    def getScaler(self):
        return self._scaler
