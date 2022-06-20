import sys

import cross_validation
import data
import knn
import evaluation


def main(argv):
    df = data.load_data(path=argv[1])
    folds = data.get_folds()
    k_list = [3] # ,5, 11, 25, 51, 75, 101]

    print('Part 1 - Classification')
    X = df.loc[:, ['t1', 't2', 'wind_speed', 'hum']].to_numpy()
    y = data.adjust_labels(df.loc[:, 'season'].to_numpy())
    X = data.add_noise(X)
    means, sample_stds = cross_validation.model_selection_cross_validation(knn.ClassificationKNN, k_list, X, y, folds,
                                                                           metric=evaluation.f1_score)
    for k, mean, sample_std in zip(k_list, means, sample_stds):
        print(f'k={k}, mean score: {mean}, std of scores: {sample_std}')
    evaluation.visualize_results(k_list, means, 'f1_score', 'Classification', path='Classification_plot.pdf')
    print()

    print('Part 2 - Regression')
    X = df.loc[:, ['t1', 't2', 'wind_speed']].to_numpy()
    y = df.loc[:, 'hum']
    X = data.add_noise(X)
    means, sample_stds = cross_validation.model_selection_cross_validation(knn.RegressionKNN, k_list, X, y, folds,
                                                                           metric=evaluation.rmse)
    for k, mean, sample_std in zip(k_list, means, sample_stds):
        print(f'k={k}, mean score: {mean}, std of scores: {sample_std}')
    evaluation.visualize_results(k_list, means, 'f1_score', 'Classification', path='Regression_plot.pdf')
    print()


if __name__ == '__main__':
    main(sys.argv)
