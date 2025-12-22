import numpy as np
from dataset import daily_ref_data, load_est_data, load_ref_data
import settings
from arg_parser import get_denoise_parser
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

def construct_matrix_x_from_ref(filename: str, rank: int,random=False) -> (list, np.ndarray, np.ndarray):
    daily_refs = daily_ref_data(filename)
    assert np.all([len(d) == 24 for d in daily_refs])
    if random:
        idx = np.random.choice(len(daily_refs), size=rank, replace=False)
        selected_columns = daily_refs[idx]
        mask = np.ones(len(daily_refs), dtype=bool)
        mask[idx] = False
    else:
        selected_columns = daily_refs[:rank]
        mask = np.ones(len(daily_refs), dtype=bool)
        mask[:rank] = False
    average_ref = [np.mean(x) for x in selected_columns.T]
    return average_ref, (selected_columns - np.array(average_ref)).T, mask

def construct_matrix_x_from_shuffle_est(filename: str,rank:int, random=False) -> (list, np.ndarray, np.ndarray):
    df = load_est_data(filename)
    daily_ests = divide_into_daily_est(df)
    assert np.all([len(d) == 24 for d in daily_ests])
    if random:
        idx = np.random.choice(len(daily_ests), size=rank, replace=False)
        selected_columns = daily_ests[idx]
        mask = np.ones(len(daily_ests), dtype=bool)
        mask[idx] = False
    else:
        selected_columns = daily_ests[:rank]
        mask = np.ones(len(daily_ests), dtype=bool)
        mask[:rank] = False
    average_ref = [np.mean(x) for x in selected_columns.T]
    return average_ref, (selected_columns - np.array(average_ref)).T, mask

def run():
    parser = get_denoise_parser()
    args = parser.parse_args()
    # print('Matrix X Rank is', args.x_rank)
    if args.x_from == 'ref':
        average_ref, matrix_x, mask = construct_matrix_x_from_ref(args.dataset,rank=args.x_rank)
    else:
        average_ref, matrix_x, mask = construct_matrix_x_from_shuffle_est(args.x_from_file,rank=args.x_rank)
    #print('average ref:', average_ref)
    #print('matrix_x:',matrix_x)
    U, singular_values, V = np.linalg.svd(matrix_x)
    theoretical_rank = cal_theoretical_rank(singular_values,rank=args.x_rank)
    # print('theoretical rank:', theoretical_rank)
    # print('singular values:', singular_values)
    #print('median of singular values',np.median(singular_values))
    Ur = U[:,:args.rank]
    df = load_est_data(args.est_from)
    daily_est = divide_into_daily_est(df)
    daily_residual = cal_daily_residual(daily_est, average_ref)

    """
    denoised_daily_est = (Ur @ Ur.T @ daily_residual.T).T + average_ref
    ref_data = load_ref_data(args.dataset)
    denoised_rmse = root_mean_squared_error(ref_data,denoised_daily_est.ravel())
    """
    denoised_daily_est = (Ur @ Ur.T @ daily_residual[mask].T).T + average_ref
    ref_data = load_ref_data(args.dataset)
    new_mask = []
    for i in np.where(mask)[0]:
        for j in range(i*24, (i+1)*24):
            new_mask.append(j)
    #denoised_rmse = root_mean_squared_error(ref_data[args.x_rank*24:],denoised_daily_est.ravel())
    denoised_rmse = root_mean_squared_error(ref_data[new_mask], denoised_daily_est.ravel())
    denoised_r2 = r2_score(ref_data[new_mask], denoised_daily_est.ravel())

    # print(f'rank={args.rank},rmse={round(denoised_rmse,3)},r2={round(denoised_r2,3)}')
    print(f'|{args.rank}|{round(denoised_rmse, 3)}|{round(denoised_r2, 3)}|')
    # print(','.join(map(str, singular_values)))
    return denoised_rmse



if __name__ == '__main__':
    run()
