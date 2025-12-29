from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from dataset import load_features_and_labels, load_dataset, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import settings


def rf_cv(X, y, n_estimators, max_features, max_depth, min_samples_split):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=splitter, scoring=['r2'])
    # r2 = "{:.4f}".format(np.average(scores['test_r2']))
    r2 = np.average(scores['test_r2'])
    # print(f'|{n_estimators}|{max_features}|{max_depth}|{min_samples_split}|{r2}|')
    return r2

def test_rf_cv():
    X, y = load_features_and_labels(settings.DatasetFile)
    y = y[settings.RefColumn].to_numpy()
    for n_estimators in range(100,3000,100):
        combination = ''
        max_r2 = 0
        for max_depth in [35,30,25,20,15,10,5]:
            for min_samples_split in [2,3,4,5,6,7,8]:
                r2 = rf_cv(X, y, n_estimators, 12, max_depth, min_samples_split)
                if r2 > max_r2:
                    max_r2 = r2
                    combination = f'|{n_estimators}|12|{max_depth}|{min_samples_split}|{round(r2,3)}|'
        print(combination)

def default_rf():
    df = load_dataset(settings.DatasetFile)
    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, settings.FeatureColumns],
        df.loc[:, [settings.RefColumn]],
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    """
    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, settings.ReducedFeatureColumns],
        df.loc[:, [settings.RefColumn]],
        train_size=0.7,
        shuffle=True,
        random_state=42
    )
    """
    y_train = y_train[settings.RefColumn].to_numpy()
    y_test = y_test[settings.RefColumn].to_numpy()
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)
    params = model.get_params()
    depths = [tree.get_depth() for tree in model.estimators_]
    # print("Max depth:", max(depths))
    # for k,v in params.items():
    #    print(f'{k}: {v}')
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    test_rmse = "{:.3f}".format(root_mean_squared_error(y_test, y_pred))
    test_r2 = "{:.3f}".format(r2_score(y_test, y_pred))
    train_rmse = "{:.3f}".format(root_mean_squared_error(y_train, y_pred_train))
    train_r2 = "{:.3f}".format(r2_score(y_train, y_pred_train))
    print(f'test r2:{test_r2}; test_rmse:{test_rmse}')
    print(f'train r2:{train_r2}; train_rmse:{train_rmse}')


if __name__ == '__main__':
    # rf_cv()
    default_rf()
    # test_rf_cv()