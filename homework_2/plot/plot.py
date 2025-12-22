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
    plt.figure(figsize=(12, 5))
    """
    plt.plot(selected_idx, selected_true, color='green', label='Reference Value')
    plt.plot(selected_idx, selected_pred, color='darkorange', label='Estimated Value')
    """
    plt.plot(selected_dates, selected_true, color='green', label='Ref. value')
    plt.plot(selected_dates, selected_pred, color='darkorange', label='Est. value')

    # scatter
    selected_test_idx = [idx for idx in selected_idx if idx in test_idx]
    selected_test_dates = [dates[idx] for idx in selected_test_idx]
    selected_pred = [y_pred[idx] for idx in selected_test_idx]
    """
    plt.scatter(
        selected_test_idx, selected_pred, color='red', marker='.',label='Est. Value From test data'
    )
    """
    plt.scatter(
        selected_test_dates, selected_pred, color='red', marker='.',label='Est. value from test data'
    )

    plt.xlabel('Timestamp')
    plt.ylabel(r'$O_3 (\mu gr/m^3)$')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'graphs/{model}_{train_percentage}_cons_temporal_curve.png')

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

def plot_daily_rmse(model:str, rmse: list, dates: np.ndarray[np.datetime64], split_idx: int):
    plt.figure(figsize=(12, 5))
    plt.plot(dates[0:split_idx], rmse[0:split_idx], color='darkgreen', label=r'Train data RMSE')
    plt.plot(dates[split_idx:], rmse[split_idx:], color='darkorange', label=r'Test data RMSE')

    plt.xlabel('Timestamp')
    plt.title('RMSE')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'graphs/not_shuffled_{model}_rmse.png')

def plot_daily_r2(model: str, r2: list, dates: np.ndarray[np.datetime64], split_idx: int):
    plt.figure(figsize=(12, 5))
    plt.plot(dates[0:split_idx], r2[0:split_idx], color='darkgreen', label=r'Train data $R^2$')
    plt.plot(dates[split_idx:], r2[split_idx:], color='darkorange', label=r'Test data $R^2$')

    plt.xlabel('Timestamp')
    plt.title(r'$R^2$')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'graphs/not_shuffled_{model}_r2.png')

def plot_daily_r2_and_rmse(
        model: str,
        r2: list,
        rmse: list,
        dates: np.ndarray[np.datetime64],
        split_idx: int
):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # -------- left: daily R2 --------
    axs[0].plot(
        dates[0:split_idx], r2[0:split_idx],
        color='darkgreen', label=r'Train data $R^2$'
    )
    axs[0].plot(
        dates[split_idx:], r2[split_idx:],
        color='darkorange', label=r'Test data $R^2$'
    )
    axs[0].set_xlabel('Timestamp')
    axs[0].set_title(r'$R^2$')
    axs[0].legend()

    # -------- right: daily RMSE --------
    axs[1].plot(
        dates[0:split_idx], rmse[0:split_idx],
        color='darkgreen', label=r'Train data RMSE'
    )
    axs[1].plot(
        dates[split_idx:], rmse[split_idx:],
        color='darkorange', label=r'Test data RMSE'
    )
    axs[1].set_xlabel('Timestamp')
    axs[1].set_title('RMSE')
    axs[1].legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'graphs/not_shuffled_{model}_r2_rmse.png')
