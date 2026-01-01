from sklearn.linear_model import LassoLarsCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataset import load_features_and_labels
import pandas as pd
import numpy as np
import settings


def non_shuffle(train_size):
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        shuffle=False
    )
    # x_train = X.loc[0:int(train_size * 4223) - 1, settings.FeatureColumns]
    # y_train = y[0:int(train_size * 4223)]
    tscv = TimeSeriesSplit(n_splits=5)
    model = make_pipeline(StandardScaler(), LassoLarsCV(cv=tscv)).fit(x_train, y_train)
    lasso = model[-1]
    feature_names = X.columns.to_numpy()
    selected_features = []
    for i in range(0,12):
        print(f'|{feature_names[i]}|{round(lasso.coef_[i],3)}|')
        if lasso.coef_[i] != 0:
            selected_features.append(feature_names[i])
    print(f'lambda: {lasso.alpha_}')
    print(f'selected features: {sorted(selected_features)}')
    coefs = dict(zip(feature_names, lasso.coef_))
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for k, v in sorted_coefs:
        print(f'{k} & {round(v,3)} \\\\')
    plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ":")
    plt.semilogx(
        lasso.cv_alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label=r"$\lambda$ CV")

    # plt.ylim(ymin, ymax)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Mean squared error")
    plt.legend()
    # plt.show()
    # plt.savefig(f"graphs/feature_selection/lasso_non_shuffle_{int(train_size*100)}.png")




if __name__ == "__main__":
    # main()
    # predict()
    non_shuffle(train_size=0.1)