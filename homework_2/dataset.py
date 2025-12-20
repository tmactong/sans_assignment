import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import typing as t
import settings


def load_dataset(filename:str) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        sep=',',
        header=0,
        usecols=settings.AllColumns,
        parse_dates=['Date'],
        date_format='%Y-%m-%d %H:%M:%S'
    )
    return df

def split_dataset(df:pd.DataFrame, train_size:float, shuffle:bool, random_state: t.Optional[int]) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, settings.FeatureColumns],
        df.loc[:, [settings.RefColumn]],
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state
    )
    train_idx = x_train.index
    test_idx = x_test.index
    return x_train, x_test, y_train, y_test, train_idx, test_idx

def get_all_days(filename:str):
    df = pd.read_csv(
        filename,
        sep=' ',
        usecols=[0],
        names=['date'],
        skiprows=1
    )
    print('\n'.join(sorted(pd.unique(df.date))))

def daily_ref_data(filename:str) -> np.ndarray:
    df = load_dataset(filename)
    ref = df.loc[:, settings.RefColumn].to_numpy()
    dates = df.loc[:, settings.DateColumn].to_numpy()
    current_date = dates[0].astype('datetime64[D]')
    daily_refs = [[]]
    for idx, np_date in enumerate(dates):
        date = np_date.astype('datetime64[D]')
        if date == current_date:
            daily_refs[-1].append(ref[idx])
        else:
            current_date = date
            daily_refs.append([ref[idx]])
    return np.array(daily_refs)

def load_est_data(filename:str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df

def load_ref_data(filename:str) -> pd.DataFrame:
    df = load_dataset(filename)
    return df[settings.RefColumn].to_numpy()


if __name__ == '__main__':
    # print(len(load_dataset('dataset/data_Hw2.csv')))
    # get_all_days('dataset/data_Hw2.csv')
    # daily_refs = daily_ref_data('dataset/data_Hw2.csv')
    df = load_est_data('est/rf_70.0.csv')
    print(df['Est.'])
