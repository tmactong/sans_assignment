import argparse


def get_cal_parser():
    parser = argparse.ArgumentParser(prog="calibrate.py", description="Calibration of LCS")
    parser.add_argument('dataset', help='Dataset file')
    parser.add_argument(
        '-m', '--model', choices=['mlr', 'rf', 'knn', 'gb'], help='Model name', required=True)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument(
        '--train_percentage', type=float, default=0.7,
        help='Percentage of dataset to split into train and test')
    # KNN parameters
    parser.add_argument('--k_neighbors', type=int, help='Number of neighbors')
    return parser


def get_cv_parser():
    parser = argparse.ArgumentParser(
        prog="cross_validate.py",
        description="Cross validation",
    )
    parser.add_argument('dataset', help='Dataset file')
    parser.add_argument(
        '-m', '--model', choices=['mlr', 'rf', 'knn', 'gb'], help='Model name', required=True)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument(
        '--train_percentage', type=float, default=0.7,
        help='Percentage of dataset to split into train and test')
    # KNN parameters
    parser.add_argument('--k_neighbors', type=int, help='Number of neighbors')
    parser.add_argument('--weights', type=str, choices=['uniform', 'distance'],
                        default='uniform',help='Weights file')
    # Random forest parameters
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_features', type=int, default=6,choices=[1,2,3,4,5,6] ,help='Max features')
    parser.add_argument('--max_depth', type=int, help='Max depth')
    parser.add_argument('--min_samples_split', type=int, default=2,help='Min samples split')
    return parser
