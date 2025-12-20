from dataset import load_est_data, load_ref_data
import pandas as pd

def main():
    knn_est = load_est_data('est/knn_70.0.csv')['Est.'].to_numpy()
    mlr_est = load_est_data('est/mlr_70.0.csv')['Est.'].to_numpy()
    rf_est = load_est_data('est/rf_70.0.csv')['Est.'].to_numpy()
    ref_data = load_ref_data('dataset/data_Hw2.csv')
    d = {'knn': knn_est,'mlr': mlr_est, 'rf': rf_est, 'ref': ref_data}
    df = pd.DataFrame(d)
    df.to_csv('est/not_shuffled_all.csv', index=False)


if __name__ == '__main__':
    main()