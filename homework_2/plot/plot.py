import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import PredictionErrorDisplay

TruncatedTimeStart = 528
TruncatedTimeEnd = 1200
#July1st = 528
#July30th = 1199

def plot_temporal_curve(model:str, train_percentage: int, y_test:pd.DataFrame, y_pred:np.ndarray):
    plt.figure(figsize=(12, 6))
    # plt.scatter(list(range(0, len(y_pred))), y_pred, color='darkorange', label='Estimated Value')
    plt.plot(list(range(0, len(y_pred))), y_pred, color='darkorange', label='Estimated Value')
    plt.plot(list(range(0, len(y_pred))),y_test.values, label='Reference Value', color='green')
    plt.ylabel(r'$O_3 (\mu gr/m^3)$')
    plt.title('Reference vs Estimated Value')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'graphs/{model}_{train_percentage}_testdata_temporal_curve.png')


def plot_continuous_temporal_curve(
        model: str, train_percentage: int,
        y_true:pd.DataFrame, y_pred:np.ndarray,
        test_idx: pd.Index, dates: pd.DataFrame
):
    # Plot from 1st July to 30th July
    selected_pred = y_pred[TruncatedTimeStart:TruncatedTimeEnd]
    selected_true = y_true[TruncatedTimeStart:TruncatedTimeEnd]
    selected_idx = list(range(TruncatedTimeStart,TruncatedTimeEnd))
    selected_dates = dates[TruncatedTimeStart:TruncatedTimeEnd]
    plt.figure(figsize=(12, 6))
    """
    plt.plot(selected_idx, selected_true, color='green', label='Reference Value')
    plt.plot(selected_idx, selected_pred, color='darkorange', label='Estimated Value')
    """
    plt.plot(selected_dates, selected_true, color='green', label='Reference Value')
    plt.plot(selected_dates, selected_pred, color='darkorange', label='Estimated Value')

    # scatter
    selected_test_idx = [idx for idx in selected_idx if idx in test_idx]
    selected_test_dates = [dates[idx] for idx in selected_test_idx]
    selected_pred = [y_pred[idx] for idx in selected_test_idx]
    """
    plt.scatter(
        selected_test_idx, selected_pred, color='red', marker='.',label='Estimated Value From test data'
    )
    """
    plt.scatter(
        selected_test_dates, selected_pred, color='red', marker='.',label='Estimated Value From test data'
    )

    plt.xlabel('Timestamp')
    plt.ylabel(r'O_3 (\mu gr/m^3)')
    plt.title('Predicted Value')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'graphs/{model}_{train_percentage}_cons_temporal_curve.png')


"""
def plot_continuous_temporal_curve(
        raw_data:pd.DataFrame, y_pred_train:np.ndarray, y_pred:np.ndarray,
        train_idx: pd.Index, test_idx: pd.Index
):
    # Plot from 1st July to 30th July
    pred = np.concatenate((y_pred_train, y_pred), axis=None)
    pred_index = np.concatenate((train_idx.to_numpy(), test_idx.to_numpy()), axis=None)
    sorted_pred = pd.DataFrame(pred, index=pred_index)
    sorted_pred.sort_index()
    raw_data = raw_data.assign(pred=sorted_pred)
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
"""



def plot_scatter(model, y_true: np.ndarray, y_pred: np.ndarray):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

    """Actual vs Predicted Value"""
    PredictionErrorDisplay.from_predictions(
        y_true,
        y_pred=y_pred,
        kind='actual_vs_predicted',
        ax=axs[0],
        random_state=0
    )
    axs[0].set_title('Actual vs Predicted Value')

    """Residual vs Predicted Value"""
    PredictionErrorDisplay.from_predictions(
        y_true,
        y_pred=y_pred,
        kind='residual_vs_predicted',
        ax=axs[1],
        random_state=0
    )
    axs[1].set_title('Residuals vs Predicted Value')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'graphs/{model}_70_residual_vs_predicted.png')

