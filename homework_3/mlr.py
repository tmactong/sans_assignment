from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from dataset import load_features_and_labels, load_selected_features_and_labels
import numpy as np
import pandas as pd
import settings
from optimal_hyper_params import non_shuffle_features_lasso, non_shuffle_features


def not_shuffle(train_size: float, model: str ,selected_features: list[str]=settings.FeatureColumns):
    X, y = load_features_and_labels(settings.DatasetFile)
    X = X.loc[:, selected_features]
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        shuffle=False
    )
    train_idx = x_train.index
    test_idx = x_test.index
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    y_pred = mlr.predict(x_test)
    y_pred_train = mlr.predict(x_train)
    test_rmse = round(root_mean_squared_error(y_test, y_pred),3)
    test_r2 = round(r2_score(y_test, y_pred),3)
    train_rmse = round(root_mean_squared_error(y_train, y_pred_train),3)
    train_r2 = round(r2_score(y_train, y_pred_train),3)

    """Overall statistics"""
    pred = np.concatenate((y_pred_train, y_pred), axis=None)
    pred_index = np.concatenate((train_idx.to_numpy(), test_idx.to_numpy()), axis=None)
    sorted_pred = pd.DataFrame(pred, index=pred_index).sort_index()
    overall_rmse = round(root_mean_squared_error(y, sorted_pred),3)
    overall_r2 = round(r2_score(y, sorted_pred),3)
    print(f"|{model}|{train_r2}|{train_rmse}|{test_r2}|{test_rmse}|{overall_r2}|{overall_rmse}|")


if __name__ == "__main__":
    train_size = 0.8
    not_shuffle(train_size=train_size, model='mlr (full features)')
    not_shuffle(train_size=train_size, model= 'mlr (sfs selected features)',selected_features=non_shuffle_features[train_size])
    not_shuffle(train_size=train_size, model='mlr (lass selected features)',selected_features=non_shuffle_features_lasso[train_size])