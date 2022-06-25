import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler


class KNN:
    def ___init___(self, k):
        """
            object instantiation, save k and define a scaler object
            :param self: an object of type KNN
            :param k: the value of k we will use to do k-NN algorithm
        """

        """ object instantiation"""
        self.stds = None  # an array with the stds of each feature.
        self.means = None  # an array with the means of each feature.

        """save k"""
        self.k = k

        """define a scaler object"""
        self.scaler = StandardScaler()

    def fit(self, X_train, Y_train):
        """
            fit scaler and save X_train and Y_train
            :param self: an object of type KNN
            :param X_train: rows of data that used to train the model
            :param Y_train: the labels of the rows X_train
        """

        """ fit scaler"""
        self.X_train = self.scaler.fit_transform(X_train)
        """save X_train and y_train"""
        self.Y_train = Y_train

    @abstractmethod
    def predict(self, X_test):
        """
            predict labels for X_test and return predicted labels. abstract method.
            :param self: an object of type KNN
        """

    def neighbours_indices(self, x):
        """
            for a given point x, find indices of k closest points in the training set
            :param self: an object of type KNN
            :param x: given point
            :return: np array containing k closest points of point x
        """

        return np.argsort(np.array([self.dist(x, train_point) for train_point in self.X_train]))[:self.k]

    @staticmethod
    def dist(x1, x2):
        """
            return Eculidean distance between x1 and x2. static method
            :param x1: first point
            :param x2: second point
            :return: eculidean distance between x1 and x2
        """

        return np.linalg.norm(x1 - x2)


class RegressionKNN(KNN):
    def __init__(self, k):
        """
            object instantiation, parent class instantiation
            :param self: an object of type KNN
            :param k: the value of k we will use to do k-NN algorithm
        """

        super().___init___(k)

    def predict(self, X_test):
        """
            predict labels for X_test and return predicted labels
            :param self: an object of type KNN
            :param X_test: rows of data that used to test the model
            :return: the predicted label by the model
        """

        X_test = self.scaler.transform(X_test)
        predicted_labels = np.zeros(X_test.shape[0])
        for point_index in range(X_test.shape[0]):
            closest_labels = np.zeros(self.k)
            closest_points_indexes = super().neighbours_indices(X_test[point_index])
            for i, index in enumerate(closest_points_indexes):
                closest_labels[i] = self.Y_train[index]
            predicted_labels[point_index] = np.average(closest_labels, weights=None)
        return predicted_labels


class ClassificationKNN(KNN):
    def __init__(self, k):
        """
            object instantiation, parent class instantiation
            :param self: an object of type KNN
            :param k: the value of k we will use to do k-NN algorithm
           """

        super().___init___(k)

    def predict(self, X_test):
        """
            predict labels for X_test and return predicted labels
            :param self: an object of type KNN
            :param X_test: rows of data that used to test the model
            :return: the predicted label by the model
        """

        X_test = self.scaler.transform(X_test)
        predicted_labels = np.zeros(X_test.shape[0], dtype=int)
        for point_index in range(X_test.shape[0]):
            closest_labels = np.zeros(self.k)
            closest_points_indexes = super().neighbours_indices(X_test[point_index])
            for i in range(self.k):
                closest_labels[i] = self.Y_train[closest_points_indexes[i]]
            predicted_labels[point_index] = stats.mode(closest_labels)[0][0]
        return predicted_labels
