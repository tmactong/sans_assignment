from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataset import load_features_and_labels
import numpy as np
import pandas as pd
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
    model = make_pipeline(StandardScaler(), Ridge())
    gscv = GridSearchCV(
        model, param_grid={'ridge__alpha': np.logspace(-3, 3, 50)},
        scoring='neg_mean_squared_error', cv=5, return_train_score=True
    )
    gscv.fit(x_train, y_train)
    best_model = gscv.best_estimator_
    ridge = best_model.named_steps['ridge']
    train_idx = x_train.index
    test_idx = x_test.index
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    y_pred_train = best_model.predict(x_train)
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
    print(f"|ridge|{train_r2}|{train_rmse}|{test_r2}|{test_rmse}|{overall_r2}|{overall_rmse}|")


def predict_non_shuffle(train_size: float):
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train = X.loc[0:int(train_size * 4223) - 1, settings.FeatureColumns]
    y_train = y[0:int(train_size * 4223)]
    x_test = X.loc[int(train_size * 4223):, settings.FeatureColumns]
    y_test = y[int(train_size * 4223):]
    model = make_pipeline(StandardScaler(), Ridge())
    gscv = GridSearchCV(
        model, param_grid={'ridge__alpha': np.logspace(-3, 3, 50)},
        scoring='r2', cv=5, return_train_score=True
    )
    gscv.fit(x_train, y_train)
    best_model = gscv.best_estimator_
    train_idx = x_train.index
    test_idx = x_test.index
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    y_pred_train = best_model.predict(x_train)
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
    print(f"|ridge|{train_r2}|{train_rmse}|{test_r2}|{test_rmse}|{overall_r2}|{overall_rmse}|")




def main():
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    model = make_pipeline(StandardScaler(), Ridge())
    gscv = GridSearchCV(
        model, param_grid={'ridge__alpha': np.logspace(-3, 3, 50)},
        scoring='neg_mean_squared_error', cv=5, return_train_score=True
    )
    gscv.fit(x_train, y_train)
    best_alpha = gscv.best_params_['ridge__alpha']
    alphas = gscv.cv_results_['param_ridge__alpha'].data
    mean_test = [abs(x) for x in gscv.cv_results_['mean_test_score']]
    print('alphas:', alphas)
    print("Mean test score:", mean_test)
    best_model = gscv.best_estimator_
    ridge = best_model.named_steps['ridge']
    print("Best alpha:", best_alpha)
    print(ridge.coef_)
    print(X.columns.to_numpy())
    feature_coefs = dict(zip(X.columns.to_numpy(), [round(abs(float(x)),3) for x in ridge.coef_]))
    sorted_feature_coefs = sorted(feature_coefs.items(), key=lambda x: x[-1],reverse=True)
    for k,v in sorted_feature_coefs:
        print(f'|{k}|{v}|')
    for i in range(0,12):
        print(f'{i+1}: {sum([x[-1] for x in sorted_feature_coefs[0:i+1]])}')
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    plt.plot(alphas, mean_test, marker='o')
    plt.vlines(best_alpha, min(mean_test),max(mean_test),
               color='black',linestyles='dashed',label=r'best $\lambda$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('mean squared error')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/feature_selection/ridge.png')


def non_shuffle():
    X, y = load_features_and_labels(settings.DatasetFile)
    x_train = X.loc[0:28 * 24 - 1, settings.FeatureColumns]
    y_train = y[0:28 * 24]
    model = make_pipeline(StandardScaler(), Ridge())
    gscv = GridSearchCV(
        model, param_grid={'ridge__alpha': np.logspace(-3, 3, 50)},
        scoring='r2', cv=5, return_train_score=True
    )
    gscv.fit(x_train, y_train)
    best_alpha = gscv.best_params_['ridge__alpha']
    alphas = gscv.cv_results_['param_ridge__alpha'].data
    mean_test = [abs(x) for x in gscv.cv_results_['mean_test_score']]
    print('alphas:', alphas)
    print("Mean test score:", mean_test)
    best_model = gscv.best_estimator_
    ridge = best_model.named_steps['ridge']
    print("Best alpha:", best_alpha)
    print(ridge.coef_)
    print(X.columns.to_numpy())
    feature_coefs = dict(zip(X.columns.to_numpy(), [round(abs(float(x)),3) for x in ridge.coef_]))
    sorted_feature_coefs = sorted(feature_coefs.items(), key=lambda x: x[-1],reverse=True)
    for k,v in sorted_feature_coefs:
        print(f'|{k}|{v}|')
    for i in range(0,12):
        print(f'{i+1}: {sum([x[-1] for x in sorted_feature_coefs[0:i+1]])}')
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    plt.plot(alphas, mean_test, marker='o')
    plt.vlines(best_alpha, min(mean_test),max(mean_test),
               color='black',linestyles='dashed',label=r'best $\lambda$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('mean squared error')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/feature_selection/ridge_non_shuffle.png')



if __name__ == "__main__":
    # main()
    # predict()
    # non_shuffle()
    predict_non_shuffle()