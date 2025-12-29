import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from dataset import load_features_and_labels, load_features_and_labels_together, load_dataset
import settings


plt.rcParams['font.family'] = 'American Typewriter'



def feature_importance():
    X,y = load_features_and_labels(settings.DatasetFile)
    ridge = RidgeCV(alphas=np.logspace(-6, 6, 5)).fit(X, y)
    importance = np.abs(ridge.coef_)
    feature_names = np.array(settings.FeatureColumns)
    plt.bar(height=importance, x=feature_names)
    plt.title("Feature importance via coefficients")
    # plt.show()
    plt.savefig("graphs/data_observation/feature_importance.png")


def cal_statistics():
    X, y = load_features_and_labels(settings.DatasetFile)
    for feature_name in settings.FeatureColumns:
        print(f'|{feature_name}|{round(np.mean(X[feature_name]),3)}|{round(np.std(X[feature_name]),3)}|')

def cal_correlation():
    X = load_features_and_labels_together(settings.DatasetFile)
    corr_matrix = X.corr()
    target_corr = corr_matrix[settings.RefColumn]
    #print(target_corr.abs().sort_values(ascending=False))
    corr_dict = {}
    for name in target_corr.index:
        corr_dict[name] = target_corr[name]
    plt.figure(figsize=(10, 5))
    plt.bar(
        height=sorted(map(abs, corr_dict.values()), reverse=True),
        x=sorted(corr_dict.keys(), key=lambda y: abs(corr_dict[y]), reverse=True),
    )
    # plt.show()
    plt.savefig("graphs/data_observation/correlation.png")


def cal_permutation_importance():
    df = load_dataset(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, settings.FeatureColumns],
        df.loc[:, [settings.RefColumn]],
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    y_train = y_train[settings.RefColumn].to_numpy()
    y_test = y_test[settings.RefColumn].to_numpy()
    model = RandomForestRegressor(n_estimators=100, max_features=12, random_state=42)
    model.fit(x_train, y_train)
    result_train = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2, scoring='r2')
    result_test = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2, scoring='r2')
    sorted_importance_idx_train = result_train.importances_mean.argsort()
    importances_train = pd.DataFrame(
        result_train.importances[sorted_importance_idx_train].T,
        columns=x_train.columns[sorted_importance_idx_train],
    )
    sorted_importance_idx_test = result_test.importances_mean.argsort()
    importances_test = pd.DataFrame(
        result_test.importances[sorted_importance_idx_test].T,
        columns=x_test.columns[sorted_importance_idx_test],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    importances_train.plot.box(vert=False, whis=10,ax=axes[0])
    axes[0].axvline(x=0, color='k', linestyle='--')
    axes[0].set_xlabel(r'Decrease in $R^2$ score')
    axes[0].set_title('Permutation Importance (Train Set)')

    importances_test.plot.box(vert=False, whis=10,ax=axes[1])
    axes[1].axvline(x=0, color='k', linestyle='--')
    axes[1].set_xlabel(r'Decrease in $R^2$ score')
    axes[1].set_title('Permutation Importance (Test Set)')
    fig.tight_layout()
    # plt.show()
    plt.savefig("graphs/data_observation/permutation_importance.png")




if __name__ == "__main__":
    # feature_importance()
    # cal_statistics()
    # cal_correlation()
    cal_permutation_importance()