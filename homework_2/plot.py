import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TruncatedTimeStart = 528
TruncatedTimeEnd = 1032
#July1st = 528
#July30th = 1199

def plot_temporal_curve(y_test:pd.DataFrame, y_pred:np.ndarray):
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='Reference Value', linewidth=2)
    plt.plot(y_pred, label='Predicted Value', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('O3')
    plt.title('Reference Value vs Predicted Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_continuous_temporal_curve(
        raw_data:pd.DataFrame, y_pred_train:np.ndarray, y_pred:np.ndarray,
        train_idx: pd.Index, test_idx: pd.Index
):
    """Plot from 1st July to 30th July"""
    pred = np.concatenate((y_pred_train, y_pred), axis=None)
    pred_index = np.concatenate((train_idx.to_numpy(), test_idx.to_numpy()), axis=None)
    sorted_pred = pd.DataFrame(pred, index=pred_index)
    sorted_pred.sort_index()
    raw_data = raw_data.assign(pred=sorted_pred)
    # raw_data.to_csv('xx.csv')
    selected_pred = raw_data.loc[TruncatedTimeStart:TruncatedTimeEnd, 'pred']
    selected_train_idx, selected_test_idx = [], []
    for idx in range(TruncatedTimeStart, TruncatedTimeEnd):
        if idx in train_idx:
            selected_train_idx.append(idx)
        else:
            selected_test_idx.append(idx)
    plt.figure(figsize=(12, 5))
    #plt.plot(list(range(TruncatedTimeStart,TruncatedTimeEnd+1)),selected_pred.values, linewidth=2)
    plt.plot(selected_train_idx, selected_pred[selected_train_idx], '.', color='red', label='train')
    plt.plot(selected_test_idx, selected_pred[selected_test_idx], '.', color='blue', label='test')
    plt.plot(raw_data.loc[TruncatedTimeStart:TruncatedTimeEnd+1, 'RefSt_O3'], label='Reference', color='black', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('O3')
    plt.title('Predicted Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_scatter():
    pass