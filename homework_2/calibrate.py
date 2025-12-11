from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from arg_parser import get_cal_parser
from plot import plot_temporal_curve, plot_continuous_temporal_curve
from dataset import load_dataset, split_dataset
import settings


def main():
    parser = get_cal_parser()
    args = parser.parse_args()
    random_state = 42 if args.shuffle else None
    df = load_dataset(args.dataset)
    """split dataset into train and test"""
    x_train, x_test, y_train, y_test, train_idx, test_idx = split_dataset(
        df, args.train_percentage, shuffle=args.shuffle, random_state=random_state)
    match args.model:
        case 'mlr':
            model = LinearRegression()
        case 'knn':
            model = KNeighborsRegressor(n_neighbors=args.k_neighbors)
        case _:
            raise NotImplementedError
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    print('##### Testing Statistics #####')
    print('RMSE:', root_mean_squared_error(y_test, y_pred))
    print('R2:', r2_score(y_test, y_pred))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print("##### Training Statistics #####")
    print('RMSE:', root_mean_squared_error(y_train, y_pred_train))
    print('R2:', r2_score(y_train, y_pred_train))
    print('MAE:', mean_absolute_error(y_train, y_pred_train))
    # plot_temporal_curve(y_test[:settings.PlotHours], y_pred[:settings.PlotHours])
    # plot_continuous_temporal_curve(df, y_pred_train, y_pred, train_idx, test_idx)


if __name__ == "__main__":
    main()
