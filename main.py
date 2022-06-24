import sys

import cross_validation
import data
import knn
import evaluation
import numpy as np


def main(argv):
    df = data.load_data(path=argv[1])
    folds = data.get_folds()
    k_list = np.array([3, 5, 11, 25, 51, 75, 101])

    print('Part1 - Classification')
    X = df.loc[:, ['t1', 't2', 'wind_speed', 'hum']].to_numpy()
    y = data.adjust_labels(df.loc[:, 'season'].to_numpy())
    X = data.add_noise(X)
    means, sample_stds = cross_validation.model_selection_cross_validation(knn.ClassificationKNN, k_list, X, y, folds,
                                                                           metric=evaluation.f1_score)
    for k, mean, sample_std in zip(k_list, means, sample_stds):
        print(f'k={k}, mean score: {mean:.4f}, std of scores: {sample_std:.4f}')
    evaluation.visualize_results(k_list, means, 'f1_score', 'Classification', path='Classification_plot.png')
    print()

    print('Part2 - Regression')
    X = df.loc[:, ['t1', 't2', 'wind_speed']].to_numpy()
    y = df.loc[:, 'hum'].to_numpy()
    X = data.add_noise(X)
    means, sample_stds = cross_validation.model_selection_cross_validation(knn.RegressionKNN, k_list, X, y, folds,
                                                                           metric=evaluation.rmse)
    for k, mean, sample_std in zip(k_list, means, sample_stds):
        print(f'k={k}, mean score: {mean:.4f}, std of scores: {sample_std:.4f}')
    evaluation.visualize_results(k_list, means, 'RMSE', 'Regression', path='Regression_plot.png')


if __name__ == '__main__':
    main(sys.argv)
