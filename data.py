import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

np.random.seed(2)


def load_data(path):
    """ reads and returns the pandas DataFrame """
    return pd.read_csv(path)


def adjust_labels(y):
    """
    adjust labels of season from {0,1,2,3} to {0,1}
    :param y: array of labels of seasons where 0=spring, 1=summer, 2=fall, 3=winter
    :return: adjusted array: 0 and 1 become 0, 2 and 3 become 1.
    """
    return np.array([0 if season == 0 or season == 1 else 1 for season in y])


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


class StandardScaler:

    def __init__(self):
        """
            object instantiation
            :param self: an object of type StandardScaler
        """

        self.stds = None  # an array with the sample stds of each feature.
        self.means = None  # an array with the means of each feature.

    def fit(self, X):
        """
            fit scaler by learning mean and sample std per feature (=per column) in X
            :param self: an object of type KNN
            :param X: the data to learn the stats from
        """

        self.means = [np.mean(X[:, i]) for i in range(X.shape[1])]
        self.stds = [np.std(X[:, i], ddof=1) for i in range(X.shape[1])]

    def transform(self, X):
        """
            transform X by subtracts its mean and divides by its std for every feature
            :param self: an object of type KNN
            :param X: the data to learn the stats from
            :return: the transformed X
        """

        return np.array([(X[:, i] - self.means[i]) / self.stds[i] for i in range(X.shape[1])]).T

    def fit_transform(self, X):
        """
            fit scaler by learning mean and std per feature, and then transform X
            :param self: an object of type KNN
            :param X: the data to learn the stats from
            :return: the transformed X
        """

        self.fit(X)
        return self.transform(X)

