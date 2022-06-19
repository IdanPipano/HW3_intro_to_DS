import numpy as np
import matplotlib.pyplot as plt


def calc_recall(y_true, y_pred):
    """
    Calculate recall for the given prediction.
    Recall = TP/(TP+FN).  (assumes: positive - 1, negative - 0)
    :param y_true: a np.array of the true labels
    :param y_pred: a np.array of the predicted labels
    :return: the recall measure of the given prediction
    """
    TP = sum([t == p and t == 1 for (t, p) in zip(y_true, y_pred)])
    TP_plus_FN = sum(y_true)  # equals the number of positive observations
    return TP / TP_plus_FN


def calc_precision(y_true, y_pred):
    """
    Calculate precision for the given prediction.
    Precision = TP/(TP+FP).  (assumes: positive - 1, negative - 0)
    :param y_true: a np.array of the true labels
    :param y_pred: a np.array of the predicted labels
    :return: the precision measure of the given prediction
    """
    TP = sum([t == p and t == 1 for (t, p) in zip(y_true, y_pred)])
    TP_plus_FP = sum(y_pred)
    return TP / TP_plus_FP


def f1_score(y_true, y_pred):
    """ returns f1_score of binary classification task with true labels y_true and predicted labels y_pred"""
    recall = calc_recall(y_true, y_pred)
    precision = calc_precision(y_true, y_pred)
    return (2*recall*precision) / (recall + precision)


def rmse(y_true, y_pred):
    """ returns RMSE of regression task with true labels y_true and predicted labels y_pred """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def visualize_results(k_list, scores, metric_name, title, path):
    """ plot a results graph for cross validation scores """
    plt.plot(x=k_list, y=scores, x_label='k', y_label=metric_name, title=title)
    plt.savefig(path=path)