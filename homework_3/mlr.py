from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from dataset import load_features_and_labels, load_selected_features_and_labels
import numpy as np
import pandas as pd
import settings
from optimal_hyper_params import non_shuffle_features_lasso, non_shuffle_features


def not_shuffle(train_size: float, selected_features: list[str]=settings.FeatureColumns):
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train = X.loc[0:int(train_size * 4223) - 1, selected_features]
    x_test = X.loc[int(train_size * 4223):, selected_features]
    y_train = y[0:int(train_size * 4223)]
    y_test = y[int(train_size * 4223):]
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
    print(f"|mlr|{train_r2}|{train_rmse}|{test_r2}|{test_rmse}|{overall_r2}|{overall_rmse}|")


if __name__ == "__main__":
    # not_shuffle(train_size=0.9)
    # not_shuffle(train_size=0.9, selected_features=non_shuffle_features[0.9])
    # not_shuffle(train_size=0.5, selected_features=non_shuffle_features_lasso[0.5])
    not_shuffle(0.16)
    not_shuffle(0.16, selected_features=['N_CPC', 'PM-1.0', 'NO2'])