from sklearn.linear_model import LassoLarsCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataset import load_features_and_labels
import pandas as pd
import numpy as np
import settings


def predict():
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    lasso = make_pipeline(StandardScaler(), LassoLarsCV(cv=20)).fit(x_train, y_train)
    train_idx = x_train.index
    test_idx = x_test.index
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    y_pred_train = lasso.predict(x_train)
    test_rmse = round(root_mean_squared_error(y_test, y_pred), 3)
    test_r2 = round(r2_score(y_test, y_pred), 3)
    train_rmse = round(root_mean_squared_error(y_train, y_pred_train), 3)
    train_r2 = round(r2_score(y_train, y_pred_train), 3)

    """Overall statistics"""
    pred = np.concatenate((y_pred_train, y_pred), axis=None)
    pred_index = np.concatenate((train_idx.to_numpy(), test_idx.to_numpy()), axis=None)
    sorted_pred = pd.DataFrame(pred, index=pred_index).sort_index()
    overall_rmse = round(root_mean_squared_error(y, sorted_pred), 3)
    overall_r2 = round(r2_score(y, sorted_pred), 3)
    print(f"|lasso|{train_r2}|{train_rmse}|{test_r2}|{test_rmse}|{overall_r2}|{overall_rmse}|")


def predict_non_shuffle(train_size:float):
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train = X.loc[0:int(train_size*4223) - 1, settings.FeatureColumns]
    y_train = y[0:int(train_size*4223)]
    x_test = X.loc[int(train_size*4223):, settings.FeatureColumns]
    y_test = y[int(train_size*4223):]
    lasso = make_pipeline(StandardScaler(), LassoLarsCV(cv=5)).fit(x_train, y_train)
    train_idx = x_train.index
    test_idx = x_test.index
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    y_pred_train = lasso.predict(x_train)
    test_rmse = round(root_mean_squared_error(y_test, y_pred), 3)
    test_r2 = round(r2_score(y_test, y_pred), 3)
    train_rmse = round(root_mean_squared_error(y_train, y_pred_train), 3)
    train_r2 = round(r2_score(y_train, y_pred_train), 3)

    """Overall statistics"""
    pred = np.concatenate((y_pred_train, y_pred), axis=None)
    pred_index = np.concatenate((train_idx.to_numpy(), test_idx.to_numpy()), axis=None)
    sorted_pred = pd.DataFrame(pred, index=pred_index).sort_index()
    overall_rmse = round(root_mean_squared_error(y, sorted_pred), 3)
    overall_r2 = round(r2_score(y, sorted_pred), 3)
    print(f"|lasso|{train_r2}|{train_rmse}|{test_r2}|{test_rmse}|{overall_r2}|{overall_rmse}|")


def main():
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    model = make_pipeline(StandardScaler(), LassoLarsCV(cv=20)).fit(x_train, y_train)
    lasso = model[-1]
    print(lasso.coef_)
    feature_names = X.columns.to_numpy()
    print(feature_names)
    for i in range(0,10):
        print(f'|{feature_names[i]}|{round(lasso.coef_[i],3)}|')
    # print(f'lambda: {lasso.alpha_}')
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
    plt.ylabel("Mean square error")
    plt.legend()
    # plt.show()
    plt.savefig("graphs/feature_selection/lasso.png")


def non_shuffle(train_size):
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train = X.loc[0:int(train_size * 4223) - 1, settings.FeatureColumns]
    y_train = y[0:int(train_size * 4223)]
    model = make_pipeline(StandardScaler(), LassoLarsCV(cv=5)).fit(x_train, y_train)
    lasso = model[-1]
    feature_names = X.columns.to_numpy()
    selected_features = []
    for i in range(0,10):
        print(f'|{feature_names[i]}|{round(lasso.coef_[i],3)}|')
        if lasso.coef_[i] != 0:
            selected_features.append(feature_names[i])
    print(f'lambda: {lasso.alpha_}')
    print(f'selected features: {selected_features}')
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
    plt.ylabel("Mean square error")
    plt.legend()
    # plt.show()
    plt.savefig("graphs/feature_selection/lasso_non_shuffle_70.png")




if __name__ == "__main__":
    # main()
    # predict()
    # non_shuffle(train_size=0.9)
    predict_non_shuffle(train_size=0.7)