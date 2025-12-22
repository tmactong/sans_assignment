from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from dataset import load_dataset, split_dataset
import settings
import sys


def run(shuffle, train_percentage, model,
        train_weeks=4,
        k_neighbors=None,weights=None,
        n_estimators=None,max_features=None,max_depth=None,min_samples_split=None
        ):
    df = load_dataset('dataset/data_Hw2.csv')
    if shuffle:
        x_train_raw, _, y_train_raw, _, _, _ = split_dataset(
            df, train_percentage, shuffle=True, random_state=42)
        x_train, _, y_train, y_test = train_test_split(
            x_train_raw,
            y_train_raw,
            train_size=settings.CrossValidationTrainPercentage,
            shuffle=True,
            random_state=42
        )
        y_train = y_train[settings.RefColumn].to_numpy()
    else:
        """use first 4 weeks data to train"""
        x_train_raw = df.loc[0:train_weeks * 7 * 24 - 1, settings.FeatureColumns]
        y_train_raw = df.loc[0:train_weeks * 7 * 24 - 1, [settings.RefColumn]]
        x_train, _, y_train, _ = train_test_split(
            x_train_raw,
            y_train_raw,
            train_size=settings.CrossValidationTrainPercentage,
            shuffle=True,
            random_state=42
        )
        y_train = y_train[settings.RefColumn].to_numpy()
    match model:
        case 'knn':
            """k nearest neighbors"""
            model = KNeighborsRegressor(
                n_neighbors=k_neighbors,
                weights=weights
            )
        case 'rf':
            """random forest regressor"""
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        case _:
            raise NotImplementedError
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=splitter,
        scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error']
    )
    #rmse = "{:.4f}".format(-1 * np.average(scores['test_neg_root_mean_squared_error']))
    #r2 = "{:.4f}".format(np.average(scores['test_r2']))
    #mae = "{:.4f}".format(-1 * np.average(scores['test_neg_mean_absolute_error']))
    # print(f'|{getattr(args, user_specified_argument())}|{rmse}|{r2}|{mae}|')
    return np.average(scores['test_r2'])


def knn():
    train_percentage = float(sys.argv[1])
    max_r2_uniform = 0
    best_n_neighbors_uniform = 0
    max_r2_distance = 0
    best_n_neighbors_distance = 0
    for i in range(1,16):
        r2 = run(True, train_percentage, 'knn', k_neighbors=i, weights='uniform')
        if r2 > max_r2_uniform:
            best_n_neighbors_uniform = i
            max_r2_uniform = r2
    for i in range(1,16):
        r2 = run(True, train_percentage, 'knn', k_neighbors=i, weights='distance')
        if r2 > max_r2_distance:
            best_n_neighbors_distance = i
            max_r2_distance = r2
    if max_r2_distance > max_r2_uniform:
        print(f'|{train_percentage}|{best_n_neighbors_distance}|distance|')
    else:
        print(f'|{train_percentage}|{best_n_neighbors_uniform}|uniform|')


def rf_shuffle():
    train_percentage = float(sys.argv[1])
    match train_percentage:
        case 0.1:
            max_n_estimators = 200
        case 0.2:
            max_n_estimators = 300
        case 0.3:
            max_n_estimators = 500
        case 0.4:
            max_n_estimators = 600
        case 0.5:
            max_n_estimators = 800
        case 0.6:
            max_n_estimators = 900
        case 0.7:
            max_n_estimators = 250
        case 0.8:
            max_n_estimators = 1100
        case 0.9:
            max_n_estimators = 1200
        case _:
            raise NotImplementedError

    r2_dict = {}
    #n_estimators = [10,20,30,40,50,60,70,80,90] + list(range(100,max_n_estimators+50,50))
    n_estimators = list(range(300,max_n_estimators+50,50))
    # max_features = [4,5,6]
    max_features = [6]
    max_depths = [None,20,25,30,35,40,45,50]
    # max_depths = [None,5,10,15,20,25]
    min_samples_splits = [2,3]
    for n_estimator in n_estimators:
        print(f'n_estimator: {n_estimator}')
        for max_feature in max_features:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    """
                    print(f'n_estimators={n_estimator},'
                             f'max_feature={max_feature},'
                             f'max_depth={max_depth},'
                             f'min_samples_split={min_samples_split}')
                    """
                    r2 = run(True, train_percentage, 'rf',
                             n_estimators=n_estimator,max_features=max_feature,
                             max_depth=max_depth,min_samples_split=min_samples_split)
                    r2_dict[(f'n_estimators={n_estimator},'
                             f'max_feature={max_feature},'
                             f'max_depth={max_depth},'
                             f'min_samples_split={min_samples_split}')]=r2
    #max_key = max(r2_dict, key=r2_dict.get)
    #max_r2 = r2_dict[max_key]
    max_r2 = max(r2_dict.values())
    max_keys = [k for k, v in r2_dict.items() if v == max_r2]
    for k in max_keys:
        print(f'{train_percentage}|{k}|{r2_dict[k]}')
    # print(train_percentage,';',max_key, max_r2)


def rf_non_shuffle():
    train_weeks = int(sys.argv[1])
    match train_weeks:
        case 1:
            max_n_estimators = 150
        case 2:
            max_n_estimators = 250
        case 3:
            max_n_estimators = 350
        case 4:
            max_n_estimators = 500
        case _:
            raise NotImplementedError

    r2_dict = {}
    n_estimators = [10,20,30,40,50,60,70,80,90] + list(range(100,max_n_estimators+50,50))
    max_features = [5,6]
    max_depths = [None,5,10,15,20,25,30,35,40,45,50]
    min_samples_splits = [2,3,4,5]
    for n_estimator in n_estimators:
        print(f'n_estimator: {n_estimator}')
        for max_feature in max_features:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    """
                    print(f'n_estimators={n_estimator},'
                             f'max_feature={max_feature},'
                             f'max_depth={max_depth},'
                             f'min_samples_split={min_samples_split}')
                    """
                    r2 = run(False, 0, 'rf',train_weeks=train_weeks,
                             n_estimators=n_estimator,max_features=max_feature,
                             max_depth=max_depth,min_samples_split=min_samples_split)
                    r2_dict[(f'n_estimators={n_estimator},'
                             f'max_feature={max_feature},'
                             f'max_depth={max_depth},'
                             f'min_samples_split={min_samples_split}')]=r2
    #max_key = max(r2_dict, key=r2_dict.get)
    #max_r2 = r2_dict[max_key]
    #print(train_weeks,';',max_key, max_r2)
    max_r2 = max(r2_dict.values())
    max_keys = [k for k, v in r2_dict.items() if v == max_r2]
    for k in max_keys:
        print(f'{train_weeks}|{k}|{r2_dict[k]}')


if __name__ == "__main__":
    # knn()
    rf_shuffle()
    # rf_non_shuffle()
