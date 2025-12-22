import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def divide_daily_metric(
        true_metrics: np.ndarray[float], temp: np.ndarray[float],
        humidity: np.ndarray[float],
        dates: np.ndarray[np.datetime64]) -> (list,list, list):
    daily_true_metric, temp_metric, humidity_metric = [[]], [[]], [[]]
    current_date = dates[0].astype('datetime64[D]')
    divided_dates = [current_date]
    for idx, npdate in enumerate(dates):
        date = npdate.astype('datetime64[D]')
        if date == current_date:
            daily_true_metric[-1].append(true_metrics[idx])
            temp_metric[-1].append(temp[idx])
            humidity_metric[-1].append(humidity[idx])
        else:
            current_date = date
            divided_dates.append(date)
            daily_true_metric.append([true_metrics[idx]])
            temp_metric.append([temp[idx]])
            humidity_metric.append([humidity[idx]])
    return ([np.mean(x) for x in daily_true_metric],
            [np.mean(x) for x in temp_metric], [np.mean(x) for x in humidity_metric],divided_dates)


def plot_daily_ts():
    df = pd.read_csv(
        '../dataset/data_Hw2.csv',
        sep=',',
        header=0,
        usecols=['Date', 'WE_O3','AE_O3','WE_NO2', 'AE_NO2', 'Temp', 'RelHum', 'RefSt_O3'],
        parse_dates=['Date'],
        date_format='%Y-%m-%d %H:%M:%S'
    )
    np_dates = df.loc[:, 'Date'].to_numpy()
    daily_y, daily_temp, daily_hum, dates = divide_daily_metric(
        df.loc[:, 'RefSt_O3'].to_numpy(),
        df.loc[:, 'Temp'].to_numpy(),
        df.loc[:, 'RelHum'].to_numpy(),
        np_dates
    )
    plt.figure(figsize=(10, 5))
    plt.plot(dates, daily_y, color='tab:orange', label=r'$O_3 (\mu gr/m^3)$')
    plt.plot(dates, daily_temp, color='tab:blue', label=r'Temperature ($\degree C$)')
    plt.plot(dates, daily_hum, color='tab:green', label=r'Relative humidity ($\%$)')

    plt.xlabel('Timestamp')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'graphs/daily_ref.png')


if __name__ == '__main__':
    plot_daily_ts()
