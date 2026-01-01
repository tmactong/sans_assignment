import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from dataset import load_features_and_labels
import settings

plt.rcParams['font.family'] = 'American Typewriter'


def sfs_mlr_non_shuffle(train_size):
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        shuffle=False
    )
    mlr = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=5)
    sfs = SFS(mlr, k_features=12, forward=True, floating=False, scoring='r2', cv=tscv)
    sfs = sfs.fit(x_train, y_train)
    subsets = sfs.subsets_
    print('\n')
    selected_features = set()
    max_score = 0
    best_features = []
    for k, v in subsets.items():
        current_selected_feature = set(v['feature_names']) - selected_features
        selected_features = set(v['feature_names'])
        if v['avg_score'] > max_score:
            best_features = selected_features
            max_score = v['avg_score']
        print(f'|{k}|{list(current_selected_feature)[0]}|{round(float(v['avg_score']), 4)}|')
    print(f'selected features: {list(sorted(best_features))}')
    selected_features = set()
    for k, v in subsets.items():
        current_selected_feature = set(v['feature_names']) - selected_features
        selected_features = set(v['feature_names'])
        print(f'{k} & {list(current_selected_feature)[0]} & {round(float(v['avg_score']), 4)} \\\\')
    # print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
    fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
    plt.grid()
    # plt.show()
    plt.savefig(f'graphs/feature_selection/sfs_mlr_non_shuffle_{int(train_size*100)}.png')


if __name__ == "__main__":
    # sfs_mlr(features=12)
    sfs_mlr_non_shuffle(train_size=0.1)