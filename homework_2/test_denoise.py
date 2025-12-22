import sys
import numpy as np
from dataset import daily_ref_data, load_est_data, load_ref_data
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score


def cal_theoretical_rank(singular_values: np.ndarray, rank: int) -> int:
    beta = 24 / float(rank)
    w = 0.56 * beta * beta * beta - 0.95 * beta * beta + 1.82 * beta + 1.43
    sigma = w * np.mean(singular_values)
    for idx, singular_value in enumerate(singular_values):
        if singular_value < sigma:
            return idx

def divide_into_daily_est(df: pd.DataFrame) -> np.ndarray:
    daily_est = []
    est = df['Est.'].to_numpy().tolist()
    days = int(len(est)/24)
    for idx in range(days):
        daily_est.append(est[idx*24:(idx+1)*24])
    return np.array(daily_est)

def cal_daily_residual(daily_est: np.ndarray, average_data: list) -> np.ndarray:
    daily_residual = []
    for daily_data in daily_est:
        daily_residual.append(daily_data - average_data)
    return np.array(daily_residual)

def construct_matrix_x_from_ref(filename: str, rank: int,random=False) -> (list, np.ndarray):
    daily_refs = daily_ref_data(filename)
    assert np.all([len(d) == 24 for d in daily_refs])
    selected_columns = daily_refs[:rank]
    average_ref = [np.mean(x) for x in selected_columns.T]
    return average_ref, (selected_columns - np.array(average_ref)).T

def construct_matrix_x_from_shuffle_est(filename: str,rank:int, random=False) -> (list, np.ndarray):
    df = load_est_data(filename)
    daily_ests = divide_into_daily_est(df)
    assert np.all([len(d) == 24 for d in daily_ests])
    selected_columns = daily_ests[:rank]
    average_ref = [np.mean(x) for x in selected_columns.T]
    return average_ref, (selected_columns - np.array(average_ref)).T

def run(x_rank,rank, x_from, x_from_file,est_from):
    dataset = 'dataset/data_Hw2.csv'
    # print('Matrix X Rank is', args.x_rank)
    if x_from == 'ref':
        average_ref, matrix_x = construct_matrix_x_from_ref(dataset,rank=x_rank)
    else:
        average_ref, matrix_x = construct_matrix_x_from_shuffle_est(x_from_file,rank=x_rank)
    #print('average ref:', average_ref)
    #print('matrix_x:',matrix_x)
    U, singular_values, V = np.linalg.svd(matrix_x)
    theoretical_rank = cal_theoretical_rank(singular_values,rank=x_rank)
    # print('theoretical rank:', theoretical_rank)
    # print('singular values:', singular_values)
    #print('median of singular values',np.median(singular_values))
    Ur = U[:,:rank]
    df = load_est_data(est_from)
    daily_est = divide_into_daily_est(df)
    daily_residual = cal_daily_residual(daily_est, average_ref)

    """
    denoised_daily_est = (Ur @ Ur.T @ daily_residual.T).T + average_ref
    ref_data = load_ref_data(args.dataset)
    denoised_rmse = root_mean_squared_error(ref_data,denoised_daily_est.ravel())
    """
    denoised_daily_est = (Ur @ Ur.T @ daily_residual[x_rank:].T).T + average_ref
    ref_data = load_ref_data(dataset)

    denoised_rmse = root_mean_squared_error(ref_data[x_rank*24:],denoised_daily_est.ravel())
    denoised_r2 = r2_score(ref_data[x_rank*24:], denoised_daily_est.ravel())

    #print(f'rank={rank},rmse={denoised_rmse}')
    return denoised_rmse, denoised_r2, theoretical_rank


def main():
    x_rank = int(sys.argv[1])
    single_run_rmses = []
    single_run_r2s = []
    for i in range(1,25):
        """X from Estimation"""
        ### not shuffled RF; x_rank=86,optimal rmse idx=16,theoretical rank=3
        # rmse,r2, t_rank = run(x_rank, i,'estimation','est/shuffled/rf_70.0.csv', 'est/not_shuffled/rf_70.0.csv')
        ### not shuffled KNN ;x_rank=105,optimal rmse idx=7,theoretical rank=3
        # rmse, r2,t_rank = run(x_rank, i, 'estimation', 'est/shuffled/knn_70.0.csv', 'est/not_shuffled/knn_70.0.csv')
        ### not shuffled MLR can't be optimized
        # rmse,r2, t_rank = run(x_rank, i, 'estimation', 'est/shuffled/mlr_70.0.csv', 'est/not_shuffled/mlr_70.0.csv')


        """X from Reference"""
        ### not shuffled RF, x_rank=92,optimal rmse idx=12,theoretical rank=3
        rmse,r2, t_rank = run(x_rank, i,'ref','', 'est/not_shuffled/rf_70.0.csv')
        ### not shuffled KNN x_rank=103,optimal rmse idx=5,theoretical rank=4
        # rmse,r2, t_rank = run(x_rank, i, 'ref', '', 'est/not_shuffled/knn_70.0.csv')
        ### not shuffled MLR, can't be optimized
        # rmse,r2, t_rank = run(x_rank, i, 'ref', '', 'est/not_shuffled/mlr_70.0.csv')


        ##### denoising shuffled training ######
        ###### Matrix X from est. data
        ### shuffled RF can't be optimized
        # rmse, r2, t_rank = run(x_rank, i, 'estimation', 'est/shuffled/rf_70.0.csv', 'est/shuffled/rf_70.0.csv')
        ### not shuffled KNN can't be optimized
        # rmse,r2, t_rank = run(x_rank, i, 'estimation', 'est/shuffled/knn_70.0.csv', 'est/shuffled/knn_70.0.csv')
        ### shuffled MLR can't be optimized
        # rmse, r2,t_rank = run(x_rank, i, 'estimation', 'est/shuffled/mlr_70.0.csv', 'est/shuffled/mlr_70.0.csv')
        ##
        ##
        ####### Matrix X from ref. data
        ### shuffled RF can't be optimized
        # rmse,r2,t_rank = run(x_rank, i,'ref','', 'est/shuffled/rf_70.0.csv')
        ### not shuffled KNN can't be optimized
        # rmse,r2, t_rank = run(x_rank, i, 'ref', '', 'est/shuffled/knn_70.0.csv')
        ### shuffled MLR can't be optimized
        # rmse,r2, t_rank = run(x_rank, i, 'ref', '', 'est/shuffled/mlr_70.0.csv')

        single_run_rmses.append(rmse)
        single_run_r2s.append(r2)
    optimal_idx = single_run_rmses.index(min(single_run_rmses))
    optimal_rmse = round(min(single_run_rmses),3)
    r2_when_optimal_rmse = round(single_run_r2s[optimal_idx],3)
    rmse_when_theoretical_rank = round(single_run_rmses[t_rank],3)
    r2_when_theoretical_rank = round(single_run_r2s[t_rank],3)
    rmse_when_rank_24 = round(single_run_rmses[-1],3)
    r2_when_rank_24 = round(single_run_r2s[-1],3)
    """
    print(
        f'x_rank={x_rank},rmse idx={single_run_rmses.index(min(single_run_rmses))+1},'
        f'theoretical rank={t_rank},rmse={round(min(single_run_rmses),3)},'
        f'r2={round(single_run_r2s[single_run_rmses.index(min(single_run_rmses))],3)},'
        f'rmse when k=24={round(single_run_rmses[-1],3)}, r2 when k=24={round(single_run_r2s[-1],3)},'
        f'rmse when t_rank={round(single_run_rmses[t_rank],3)}, r2 when t_rank={round(single_run_r2s[t_rank],3)}')
    """
    print(f'|{x_rank}|{optimal_idx+1}|{optimal_rmse}|{r2_when_optimal_rmse}|'
          f'{t_rank}|{rmse_when_theoretical_rank}|{r2_when_theoretical_rank}|'
          f'{rmse_when_rank_24}|{r2_when_rank_24}|')



if __name__ == '__main__':
    main()