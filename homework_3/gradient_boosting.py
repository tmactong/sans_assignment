import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd
import numpy as np
from dataset import load_features_and_labels, load_selected_features_and_labels
import settings
from optimal_hyper_params import non_shuffle_hyper_params, non_shuffle_features, non_shuffle_features_lasso


def boosting_iterations():
    """
    gradient boosting
    default hyperparameters:
    n_estimators=100, max_depth=3, min_samples_split=2, learning_rate=0.1
    """
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    gb = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=5,
        min_samples_split=2,
        learning_rate=0.05,
        random_state=42
    )
    gb.fit(x_train, y_train)
    y_pred = gb.predict(x_test)
    test_score = np.zeros((1000,), dtype=np.float64)
    for i, y_pred in enumerate(gb.staged_predict(x_test)):
        test_score[i] = root_mean_squared_error(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(1000) + 1,
        gb.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(1000) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


def exhaustive_search_shuffle():
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    gb = GradientBoostingRegressor(random_state=42)
    gscv = GridSearchCV(
        gb,
        param_grid={
            'n_estimators': [390, 400, 410],
            'max_depth': [5],
            'min_samples_split': [2],
            'learning_rate': [0.04, 0.05, 0.06]
        },
        scoring='r2',
        cv=5,
        return_train_score=True,
        verbose=2
    )
    gscv.fit(x_train, y_train)
    # best_model = gscv.best_estimator_
    best_params = gscv.best_params_
    print(f'best_params: {best_params}')
    results = pd.DataFrame(gscv.cv_results_)
    summary = results[
        [
            "param_n_estimators",
            "param_max_depth",
            "param_min_samples_split",
            "param_learning_rate",
            "mean_test_score",
            "std_test_score",
        ]
    ].sort_values("mean_test_score", ascending=False)
    print(summary.head())


non_shuffled_1st_gb_params = {
    'n_estimators': [200,250,300],
    'max_depth': [3,4,5],
    'min_samples_split': [3,4,5],
    'learning_rate': [0.01,0.05,0.1],
}

def exhaustive_search_non_shuffle(train_size):
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train = X.loc[0: int(4223*train_size) - 1, settings.FeatureColumns]
    y_train = y[0: int(4223*train_size)]
    gb = GradientBoostingRegressor(random_state=42)
    gscv = GridSearchCV(
        gb,
        param_grid=non_shuffled_1st_gb_params,
        scoring='r2',
        cv=5,
        return_train_score=True,
        verbose=2
    )
    gscv.fit(x_train, y_train)
    best_params = gscv.best_params_
    print(f'best_params: {best_params}')


def non_shuffle(train_size: float,selected_features: list[str]=settings.FeatureColumns):
    X, y = load_selected_features_and_labels(settings.DatasetFile, selected_features)
    x_train = X.loc[0:int(4223*train_size) - 1, selected_features]
    print(x_train.head())
    x_test = X.loc[int(4223*train_size):, selected_features]
    y_train = y[0:int(4223*train_size)]
    y_test = y[int(4223*train_size):]
    train_idx = x_train.index
    test_idx = x_test.index
    gb = GradientBoostingRegressor(
        **non_shuffle_hyper_params[train_size],
        random_state=42
    )
    gb.fit(x_train, y_train)
    y_pred = gb.predict(x_test)
    y_pred_train = gb.predict(x_train)
    """ Test statistics """
    test_rmse = round(root_mean_squared_error(y_test, y_pred), 3)
    test_r2 = round(r2_score(y_test, y_pred), 3)

    """ Train statistics """
    train_rmse = round(root_mean_squared_error(y_train, y_pred_train), 3)
    train_r2 = round(r2_score(y_train, y_pred_train), 3)

    """Overall statistics"""
    pred = np.concatenate((y_pred_train, y_pred), axis=None)
    pred_index = np.concatenate((train_idx.to_numpy(), test_idx.to_numpy()), axis=None)
    sorted_pred = pd.DataFrame(pred, index=pred_index).sort_index()
    overall_rmse = round(root_mean_squared_error(y, sorted_pred), 3)
    overall_r2 = round(r2_score(y, sorted_pred), 3)
    print(f"|{train_size}|{train_r2}|{train_rmse}|{test_r2}|{test_rmse}|{overall_r2}|{overall_rmse}|")


if __name__ == "__main__":
    # boosting_iterations()
    # exhaustive_search_shuffle()
    # main()
    # exhaustive_search_non_shuffle(train_size=0.3)
    # non_shuffle(train_size=0.3, selected_features=non_shuffle_features[0.3])
    non_shuffle(train_size=0.9)
    # non_shuffle(train_size=0.3, selected_features=non_shuffle_features_lasso[0.3])
