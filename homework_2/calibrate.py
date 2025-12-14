from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from arg_parser import get_cal_parser
from plot.plot import plot_continuous_temporal_curve, plot_scatter, plot_daily_data
from dataset import load_dataset, split_dataset
import numpy as np
import pandas as pd
import settings


def main():
    parser = get_cal_parser()
    args = parser.parse_args()
    df = load_dataset(args.dataset)
    """split dataset into train and test"""
    if args.shuffle:
        x_train, x_test, y_train, y_test, train_idx, test_idx = split_dataset(
            df, args.train_percentage, shuffle=True, random_state=42)
        y_train = y_train[settings.RefColumn].to_numpy()
    else:
        """use first 4 weeks data to train"""
        x_train = df.loc[0:args.train_data_weeks * 7 * 24 - 1, settings.FeatureColumns]
        y_train = df.loc[0:args.train_data_weeks * 7 * 24 - 1, settings.RefColumn].to_numpy()
        x_test = df.loc[args.train_data_weeks * 7 * 24:, settings.FeatureColumns]
        y_test = df.loc[args.train_data_weeks * 7 * 24:, settings.RefColumn].to_numpy()
        train_idx = x_train.index
        test_idx = x_test.index
    match args.model:
        case 'mlr':
            """multiple linear regression"""
            model = LinearRegression()
        case 'knn':
            """k nearest neighbors"""
            model = KNeighborsRegressor(
                n_neighbors=args.k_neighbors,
                weights=args.weights
            )
        case 'rf':
            """random forest regressor"""
            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_features=args.max_features,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                random_state=42
            )
        case _:
            raise NotImplementedError
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    test_rmse = "{:.3f}".format(root_mean_squared_error(y_test, y_pred))
    test_r2 = "{:.3f}".format(r2_score(y_test, y_pred))
    test_mae = "{:.3f}".format(mean_absolute_error(y_test, y_pred))
    train_rmse = "{:.3f}".format(root_mean_squared_error(y_train, y_pred_train))
    train_r2 = "{:.3f}".format(r2_score(y_train, y_pred_train))
    train_mae = "{:.3f}".format(mean_absolute_error(y_train, y_pred_train))

    """Overall statistics"""
    pred = np.concatenate((y_pred_train, y_pred), axis=None)
    pred_index = np.concatenate((train_idx.to_numpy(), test_idx.to_numpy()), axis=None)
    sorted_pred = pd.DataFrame(pred, index=pred_index).sort_index()
    y_true = df[settings.RefColumn].to_numpy()
    overall_rmse = "{:.3f}".format(root_mean_squared_error(y_true, sorted_pred))
    overall_r2 = "{:.3f}".format(r2_score(y_true, sorted_pred))
    overall_mae = "{:.3f}".format(mean_absolute_error(y_true, sorted_pred))

    if args.shuffle:
        print(f'|{args.model}|{args.train_percentage * 100}%|'
            f'{train_r2}|{train_rmse}|{train_mae}|'
            f'{test_r2}|{test_rmse}|{test_mae}|'
            f'{overall_r2}|{overall_rmse}|{overall_mae}|')
    else:
        print(f'|{args.model}|{args.train_data_weeks} weeks|'
              f'{train_r2}|{train_rmse}|{train_mae}|'
              f'{test_r2}|{test_rmse}|{test_mae}|'
              f'{overall_r2}|{overall_rmse}|{overall_mae}|')
    if args.plot:
        if args.shuffle:
            plot_continuous_temporal_curve(
                args.model, int(args.train_percentage * 100),
                df[settings.RefColumn].to_numpy(), sorted_pred[0].to_numpy(), test_idx,
                df[settings.DateColumn]
            )
            plot_scatter(args.model,df[settings.RefColumn].to_numpy(), sorted_pred[0].to_numpy())
        else:
            rmse = root_mean_squared_error(y_test, y_pred, multioutput='raw_values')
            r2 = r2_score(y_test, y_pred, multioutput='raw_values')
            plot_daily_data(rmse, r2, df.loc[args.train_data_weeks * 7 * 24:, settings.DateColumn].to_numpy())



if __name__ == "__main__":
    main()
