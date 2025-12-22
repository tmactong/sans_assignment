import argparse

def get_denoise_parser():
    parser = argparse.ArgumentParser(prog='denoise.py', description='Denoise the LCS data')
    parser.add_argument('dataset', help='Dataset file')
    parser.add_argument('--rank', type=int,choices=list(range(1,25)),help='r-rank approximation')
    parser.add_argument('--x_from', choices=['ref', 'estimation'],
                        help='construct matrix X from reference data or estimation')
    parser.add_argument('--x_from_file', help='construct matrix X from source')
    parser.add_argument('--est_from',type=str,help='estimated data source')
    parser.add_argument('--x_rank', type=int,help='Matrix X rank')
    return parser

def get_cal_parser():
    parser = argparse.ArgumentParser(prog="calibrate.py", description="Calibration of LCS")
    parser.add_argument('dataset', help='Dataset file')
    parser.add_argument(
        '-m', '--model', choices=['mlr', 'rf', 'knn', 'gb'], help='Model name', required=True)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument('--train_data_weeks', type=int, choices=[1,2,3,4],
                        default=4,help='Number of training weeks')
    parser.add_argument('--plot', action='store_true', help='Plot figure')
    parser.add_argument('--store', action='store_true', help='store estimated data')
    parser.add_argument(
        '--train_percentage', type=float, default=0.7,
        help='Percentage of dataset to split into train and test')
    # KNN parameters
    parser.add_argument('--k_neighbors', type=int, help='Number of neighbors')
    parser.add_argument('--weights', type=str, choices=['uniform', 'distance'],
                        default='uniform', help='Weights for neighbors')
    # Random forest parameters
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_features', type=int, default=6, choices=[1, 2, 3, 4, 5, 6], help='Max features')
    parser.add_argument('--max_depth', type=int, help='Max depth')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Min samples split')
    return parser


def get_cv_parser():
    parser = argparse.ArgumentParser(
        prog="cross_validate.py",
        description="Cross validation",
    )
    parser.add_argument('dataset', help='Dataset file')
    parser.add_argument(
        '-m', '--model', choices=['rf', 'knn'], help='Model name', required=True)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument(
        '--train_percentage', type=float, default=0.7,
        help='Percentage of dataset to split into train and test')
    parser.add_argument('--weeks', type=int, choices=[1,2,3,4],help='Number of weeks to train')
    # KNN parameters
    parser.add_argument('--k_neighbors', type=int, help='Number of neighbors')
    parser.add_argument('--weights', type=str, choices=['uniform', 'distance'],
                        default='uniform',help='Weights for neighbors')
    # Random forest parameters
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_features', type=int, default=6,choices=[1,2,3,4,5,6] ,help='Max features')
    parser.add_argument('--max_depth', type=int, help='Max depth')
    parser.add_argument('--min_samples_split', type=int, default=2,help='Min samples split')
    return parser
