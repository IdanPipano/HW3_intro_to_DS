import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler


class KNN:
    def ___init___(self, k):
        """ object instantiation"""
        self.stds = None  # an array with the stds of each feature.
        self.means = None  # an array with the means of each feature.

        """save k"""
        self.k = k

        """define a scaler object"""
        self.scaler = StandardScaler()

    def fit(self, X_train, Y_train):
        """ fit scaler"""
        self.X_train = self.scaler.fit_transform(X_train)
        """save X_train and y_train"""
        self.Y_train = Y_train

    @abstractmethod
    def predict(self, X_test):
        """predict labels for X_test and return predicted labels"""

    def neighbours_indices(self, x):
        """for a given point x, find indices of k closest points in the training set"""
        return sorted(np.arange(self.X_train.shape[0]), key=lambda i: self.dist(x, self.X_train[i]))[:self.k]

    def alternative_neighbours_indices(self, x):
        """for a given point x, find indices of k closest points in the training set"""
        closest_k_points_indexes = []
        for i in range(self.X_train.shape[0]):
            if i < self.k:
                closest_k_points_indexes.append(i)
            else:
                temp_point_index = i
                for j in range(len(closest_k_points_indexes)):
                    if self.dist(x, self.X_train[j]) > self.dist(x, self.X_train[temp_point_index]):
                        index_saver = closest_k_points_indexes[j]
                        closest_k_points_indexes[j] = temp_point_index
                        temp_point_index = index_saver
        return closest_k_points_indexes

        """distance_array = np.array(X_train.shape[1])
        for each i in range(distance_array.shape[0]):
            distance_array[i] = dist(X_train[i,:],x)

        lowest_dist_index
        for j in range(k):
            lowest_dist = np.amin(distance_array)
            lowest_dist_index =
        scipy.stats.mode()"""

    @staticmethod
    def dist(x1, x2):
        """return Eculidean distance between x1 and x2"""
        return np.linalg.norm(x1 - x2)


class RegressionKNN(KNN):
    def __init__(self, k):
        """object instantiation, parent class instantiation"""
        super().___init___(k)

    def predict(self, X_test):
        """predict labels for X_test and return predicted labels"""
        predicted_labels = np.zeros(X_test.shape[0])
        for point_index in range(X_test.shape[0]):
            closest_labels = np.zeros(self.k)
            closest_points_indexes = super().neighbours_indices(X_test[point_index])
            for i in range(self.k):
                closest_labels[i] = self.Y_train[closest_points_indexes[i]]
            predicted_labels[point_index] = np.average(closest_labels, weights=None)
        return predicted_labels


            


class ClassificationKNN(KNN):
    def __init__(self, k):
        """object instantiation, parent class instantiation"""
        super().___init___(k)

    def predict(self, X_test):
        """predict labels for X_test and return predicted labels"""
        X_test = self.scaler.transform(X_test)
        predicted_labels = np.zeros(X_test.shape[0], dtype=int)
        for point_index in range(X_test.shape[0]):
            closest_labels = np.zeros(self.k)
            closest_points_indexes = super().neighbours_indices(X_test[point_index])
            for i in range(self.k):
                closest_labels[i] = self.Y_train[closest_points_indexes[i]]
            predicted_labels[point_index] = stats.mode(closest_labels)[0][0]
        return predicted_labels
