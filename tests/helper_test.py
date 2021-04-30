# UNIT TESTS
import numpy as np
import sys
import math
sys.path.append("../../../..")
from neuralNet import *

def testGetF1Score():
    
    prediction = np.array([[0,0,0,1], [0,1,0,0], [1,0,0,0], [0,0,0,1], [0,1,0,0]])
    ground_truth = np.array([[0,0,0,1], [1,0,0,0], [1,0,0,0], [0,0,1,0], [0,1,0,0]])

    f1Score = getF1Score(prediction, ground_truth)
    # 1 1 0 1 / 1.5 1.5 -1 1.5 = 2/3 2/3 -0 2/3
    expectedF1Score = np.array([2/3., 2/3., 0, 2/3.])

    assert np.all(f1Score == expectedF1Score)

def testRecall():
    prediction = np.array([[0,0,0,1], [0,1,0,0], [1,0,0,0], [0,0,0,1], [0,1,0,0]])
    ground_truth = np.array([[0,0,0,1], [1,0,0,0], [1,0,0,0], [0,0,1,0], [0,1,0,0]])

    recall = getRecall(prediction, ground_truth)    
    expectedRecall = np.array([0.5, 1, 0, 1])

    assert np.all(recall == expectedRecall)

def testPrecision():
    
    prediction = np.array([[0,0,0,1], [0,1,0,0], [1,0,0,0], [0,0,0,1], [0,1,0,0]])
    ground_truth = np.array([[0,0,0,1], [1,0,0,0], [1,0,0,0], [0,0,1,0], [0,1,0,0]])

    accuracy = getPrecision(prediction, ground_truth)
    expectedAccuracy = np.array([1, 0.5, 0, 0.5])

    assert np.all(accuracy == expectedAccuracy)

if __name__ == "__main__":
    testRecall()
    testPrecision()
    tF1Score()