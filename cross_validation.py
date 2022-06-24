import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """
    run cross validation on X and y with specific model by given folds. Evaluate by given metric.
    :param model: an object (who has already been instantiated) of a knn model
    :param X: a matrix whose rows are observations and columns are features.
    :param y: an array in the length of the number of observations with a label for each observation
    :param folds: a sklearn KFold object, the output of the function data.get_folds
    :param metric: a function that receives 2 arguments y_true, y_pred, and returns a scalar
    :return: a list in the length of the number of folds (5)
    """
    list_of_metric_for_each_fold = []
    for train_indices, validation_indices in folds.split(X):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_validation = X[validation_indices]
        y_prediction_true = y[validation_indices]
        model.fit(X_train, y_train)
        y_validation_predict = model.predict(X_validation)
        list_of_metric_for_each_fold.append(metric(y_prediction_true, y_validation_predict))
    return list_of_metric_for_each_fold


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    """
    run cross validation on X and y for every model induced by values from k_list by given folds.
    Evaluate each model by given metric.
    :param model: aa class of a model
    :param k_list: a list of possible values for k
    :param X: a matrix whose rows are observations and columns are features.
    :param y: an array in the length of the number of observations with a label for each observation
    :param folds: a sklearn KFold object, the output of the function data.get_folds
    :param metric: a function that receives 2 arguments y_true, y_pred, and returns a scalar
    :return: a list in the length of k_list, where each entry is the mean of the metrics for every fold.
            and a list in the length of k_list, where each entry is the sample std of the metrics for every fold.
    """
    means = np.zeros_like(k_list, dtype=float)
    sample_stds = np.zeros_like(k_list, dtype=float)
    for i, k in enumerate(k_list):
        current_model = model(k)
        list_of_metric_for_each_fold = np.array(cross_validation_score(current_model, X, y, folds, metric))
        means[i] = np.mean(list_of_metric_for_each_fold)
        sample_stds[i] = np.std(list_of_metric_for_each_fold, ddof=1)

    return means, sample_stds



