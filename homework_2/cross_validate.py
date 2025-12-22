from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from arg_parser import get_cv_parser
from dataset import load_dataset, split_dataset
import settings
import sys
import typing as t


def user_specified_argument() -> t.Optional[str]:
    for argument in settings.CVKNNArguments + list(reversed(settings.CVRFArguments)):
        if f'--{argument}' in sys.argv:
            return argument

def main():
    parser = get_cv_parser()
    args = parser.parse_args()
    df = load_dataset(args.dataset)
    if args.shuffle:
        x_train_raw, _, y_train_raw, _, _, _ = split_dataset(
            df, args.train_percentage, shuffle=True, random_state=42)
        x_train, _, y_train, _ = train_test_split(
            x_train_raw,
            y_train_raw,
            train_size=settings.CrossValidationTrainPercentage,
            shuffle=True,
            random_state=42
        )
        y_train = y_train[settings.RefColumn].to_numpy()
    else:
        """use first x weeks data to train"""
        x_train_raw = df.loc[0:args.weeks*7*24-1, settings.FeatureColumns]
        y_train_raw = df.loc[0:args.weeks*7*24-1, [settings.RefColumn]]
        x_train, _, y_train, _ = train_test_split(
            x_train_raw,
            y_train_raw,
            train_size=settings.CrossValidationTrainPercentage,
            shuffle=True,
            random_state=42
        )
        y_train = y_train[settings.RefColumn].to_numpy()
    match args.model:
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
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=splitter,
        scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error']
    )
    #rmse = "{:.4f}".format(-1 * np.average(scores['test_neg_root_mean_squared_error']))
    r2 = "{:.4f}".format(np.average(scores['test_r2']))
    #mae = "{:.4f}".format(-1 * np.average(scores['test_neg_mean_absolute_error']))
    # print(f'|{getattr(args, user_specified_argument())}|{rmse}|{r2}|{mae}|')
    # print(f'|{getattr(args, user_specified_argument())}|{r2}|')
    print(f'|{args.n_estimators}|{r2}|')


if __name__ == "__main__":
    main()
