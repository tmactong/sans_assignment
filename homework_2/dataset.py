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



if __name__ == '__main__':
    print(len(load_dataset('dataset/data_Hw2.csv')))
    # get_all_days('dataset/data_Hw2.csv')
