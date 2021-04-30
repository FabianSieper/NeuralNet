import numpy as np
import sys
from digitDetector import train_digit_detector, getMnistData


np.seterr(all="raise")
np.random.seed(seed=3)
np.set_printoptions(threshold=sys.maxsize)
def main():


    images_train, labels_train, images_validation, labels_validation = getMnistData()
    train_digit_detector(images_train, labels_train, images_validation, labels_validation)



main()