import numpy as np
import h5py
np.random.seed(1337)


class MinMaxNormal(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''
    def __init__(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def __init__(self, X, Y):
        print("Xmin:",X.min(),"Xmax:",X.max(),"Ymin:",Y.min(),"Ymax:",Y.max())
        self._min = (X.min() if X.min() < Y.min() else Y.min())
        self._max = (X.max() if X.max() > Y.max() else Y.max())
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

    def maxmin(self):
        return self._max-self._min

    def rmse_transform(self, x):
        return x * self.maxmin()/2.



class Standard(object):

    def __init__(self):
        pass

    def fit(self, X):
        self.std = np.std(X)
        self.mean = np.mean(X)
        print("std:", self.std, "mean:", self.mean)

    def transform(self, X):
        X = 1. * (X - self.mean) / self.std
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = X * self.std + self.mean
        return X

    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean


def preprocess_normalize(spatio_data_path, temporal_data_path, channel):
    spatio_data = h5py.File(spatio_data_path)['data'][:, channel]
    temporal_data = h5py.File(temporal_data_path)['data'][:, channel]

    normal_st = MinMaxNormal(spatio_data, temporal_data)

    return normal_st.transform(spatio_data), normal_st.transform(temporal_data), normal_st