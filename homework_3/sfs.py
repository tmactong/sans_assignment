import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from dataset import load_features_and_labels
import settings

plt.rcParams['font.family'] = 'American Typewriter'


def main(features):
    X,y  = load_features_and_labels(settings.DatasetFile)
    rf = RandomForestRegressor(random_state=42)
    sfs = SFS(rf, k_features=features, forward=True,floating=False,scoring='r2',cv=5)
    sfs = sfs.fit(X, y)
    print(sfs.subsets_)
    print('\nSequential Forward Selection (k=8):')
    print(sfs.k_feature_idx_)
    print('\nBest subset (k=8):')
    print(sfs.k_feature_names_)
    print('CV Score:')
    print(sfs.k_score_)
    print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
    fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
    plt.grid()
    plt.show()
    # plt.savefig('graphs/feature_selection/sfs_k=12.png')


if __name__ == "__main__":
    main(features=7)