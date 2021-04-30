import matplotlib.pyplot as plt
import numpy as np


def visualizeHistory(values, label = "", names = None, path = None, show = True):
    """ takes:
            - values: values which shall be visualized (can also be list) -> [float] or [np.ndarray]
            - label: caption of diagram -> string
            - names: names of corresponding values -> string
            - path: where the diagram shall be stored -> string
            - show: shall the diagrams be shown after training? -> bool"""
    
    fig, ax = plt.subplots(figsize=(5,5))

    if names != None:
        ax.plot(values)
        ax.legend(names)
    else:
        ax.plot(values, label=label)
        

    ax.set_xlabel("iterations")
    ax.set_title(label)
    fig.legend()

    if path != None:
        plt.savefig(path+label+".png")

    if show:
        plt.show()




def getAccuracy(h, y):
    
    """ does:
            - computes the prediction accuracy onbase of a given prediction matrix
        takes: 
            - h: matrix of softmax values (prediction)
            - y: either a np.array of one-hot vectors or a 1-D np.array of actual numbers 
        returns:
            - float: accuracy """

    # check if the solution of the prediction is an array or a matrix
    oneHotSolution = len(y.shape) == 2

    # replace each line of the prediction with the index of the biggest element 
    # (index describes the predicted number)
    predictedNumbers = None
    amountCorrect = 0

    if oneHotSolution:

        # extract predicted numbers from the index with the biggest value (index 2 would mean, that number 2 was predicted)
        predictedNumbers = np.array([line.argmax() for line in h])   
        # transform one-hot-vectors of solution into actual numbers [0-9] using the same method as above)
        solutionNumbers = np.array([line.argmax() for line in y])

        correctPrediction = (predictedNumbers == solutionNumbers)

        amountCorrect = correctPrediction.sum()


    else:
        # if solution y is a np.array (1-D vector)
        predictedNumbers = np.array([line.argmax() for line in h.T])    

        # count the amount of correct predictions 
        amountCorrect = (predictedNumbers == y).sum()

    # compute the actual accuracy
    accuracy = amountCorrect / len(predictedNumbers)

    return accuracy


def getPrecision(h, y):
    """ does:
            - computes the prediction precision on base of a given prediction matrix
        takes: 
            - h: matrix of softmax values (prediction)
            - y: a np.array of one-hot vectors 
        returns:
            - precisions: np.ndarry of float """

    
    # extract predicted numbers from the index with the biggest value (index 2 would mean, that number 2 was predicted)
    predictedNumbers = np.array([line.argmax() for line in h])   
    # transform one-hot-vectors of solution into actual numbers [0-9] using the same method as above)
    solutionNumbers = np.array([line.argmax() for line in y])

    # amount possible ground_truths 
    amount_groundTruth = y.shape[1]
    TP = np.zeros(amount_groundTruth)
    FP = np.zeros(amount_groundTruth)

    for i in range(len(predictedNumbers)):

        if predictedNumbers[i] == solutionNumbers[i]:
            TP[predictedNumbers[i]] += 1
        else:
            FP[predictedNumbers[i]] += 1
    
    totalPredPos = TP + FP
    totalPredPos = [-1 if predPos == 0 else predPos for predPos in totalPredPos]

    return TP / totalPredPos

def getF1Score(h, y):
    """ does:
        - computes the f1-score
    takes: 
        - h: matrix of softmax values (prediction)
        - y: one-hot-vectors -> np.ndarray 
    returns:
        - list of f1 scores (one for each possible ground_truth) """

    precision = getPrecision(h,y)
    recall = getRecall(h,y)

    divisor = (precision + recall)
    # making sure not division by 0 is done ... replace all 0's with -1
    divisor = [-1 if divi == 0 else divi for divi in divisor]

    return 2 * precision * recall / divisor

def getRecall(h, y):
    """ does:
        - computes the recall of the given prediction h
    takes: 
        - h: matrix of softmax values (prediction)
        - y: one-hot-vectors -> np.ndarray 
    returns:
        - list of recalls (one for each possible ground_truth) """

    # extract predicted numbers from the index with the biggest value (index 2 would mean, that number 2 was predicted)
    predictedNumbers = np.array([line.argmax() for line in h])   
    # transform one-hot-vectors of solution into actual numbers [0-9] using the same method as above)
    solutionNumbers = np.array([line.argmax() for line in y])

    # amount possible ground_truths 
    amount_groundTruth = y.shape[1]
    # compute list for counting TP
    TP = np.zeros(amount_groundTruth)
    FN = np.zeros(amount_groundTruth)
    
    # compute TP and FN for each of the possible ground_truth's
    for i in range(len(predictedNumbers)):
        if predictedNumbers[i] == solutionNumbers[i]:
            TP[predictedNumbers[i]] += 1
        else:
            FN[solutionNumbers[i]] += 1

    actualPos = TP + FN
    actualPos = [-1 if pos == 0 else pos for pos in actualPos]
    recall = TP / actualPos

    return recall

def computeAllPerformanceStats(h, y):
    """ does:
        - computes accuracy, f1-score, recall, precision
    takes: 
        - h: matrix of softmax values (prediction)
        - y: one-hot-vectors -> np.ndarray 
    returns:
        - accuracy -> float
        - f1-score -> np.ndarray
        - recall -> np.ndarray
        - precision -> np.ndarray """


    accuracy = getAccuracy(h, y)
    precision = getPrecision(h, y)
    recall = getRecall(h, y)

    # computing f1-score
    divisor = (precision + recall)
    # making sure not division by 0 is done ... replace all 0's with -1
    divisor = [-1 if divi == 0 else divi for divi in divisor]
    f1 = 2 * precision * recall / divisor
    
    return accuracy, f1, recall, precision

def computeAccuracy(x, y, thetas, biases, scaler):
    
    """ does:
            - takes input values for a nn
            - creates predictions by using the thetas, and biases
            - computes the prediction accuracy of the given nn
        takes:
            - x: matrix of input values 
            - y: vector of solutions (e.g [1,2,3,5,3,5])
            - thetas: weights for the NN
            - biases: baises for the NN
            - scaler: a scaler, which was already fitted on the trainings data
        returns:
            - float: accuracy """
    
    transformedX = scaler.transform(x)
    a, _ = Forwardpass(transformedX, y, thetas, biases)

    accuracy = getAccuracy(a[-1], y)
   
    return accuracy


def createOnHotVectors(values, amount_values):
    """ takes:
        - values: array of integer-values between 0 and (amount_values - 1)-> np.ndarray or list
        - amount_values: the range of values contained in 'values' -> int
    returns:
        - np.ndarray of one-hot vectors """

    if type(values) == list:
        return np.identity(amount_values)[np.array(values)]

    return np.identity(amount_values)[values]
