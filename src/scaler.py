from abc import ABC
import numpy as np

class Scaler(ABC):

    def __init__(self):
        self._isFitted = False

    def fit(self, X):
        pass
    def transform(self, X):
        pass
    def inverse_transform(self, X_scaled):
        pass
    def isFitted(self):
        return self._isFitted

class NormalScaler(Scaler):
    # remember that we never want to normalize the bias column; `has_bias_column` is
    # a parameter we can set to avoid transforming the first column, to make our
    # lives easier later :-)
    def fit(self, X):

        print("INFO - Fitting scaler on data ...")

        self.min = np.nanmin(X)
        self.max = np.nanmax(X) - self.min
        self._isFitted = True


    def transform(self, X):
        return (X - self.min) / self.max
    

    def inverse_transform(self, X_scaled):
        return X_scaled * self.max + self.min

class StandardScaler(Scaler):

    def fit(self, X):
        print("INFO - Fitting scaler on data ...")

        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)

        # the standard deviation can be 0, which provokes
        # devision-by-zero errors; let's omit that:
        self.std[self.std == 0] = 0.00001
        self._isFitted = True


    def transform(self, X):
        np.seterr(all='raise')
        solution = (X - self.mean) / self.std
        np.seterr(all='warn')
        return solution

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean