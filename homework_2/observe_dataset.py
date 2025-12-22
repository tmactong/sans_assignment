from dataset import load_dataset
import numpy as np
import settings


def y_true_metric():
    df = load_dataset('dataset/data_Hw2.csv')
    y_true_mean = np.mean(df[settings.RefColumn])
    y_true_var = np.std(df[settings.RefColumn])
    print(round(y_true_mean, 3))
    print(round(y_true_var, 3))

def main():
    pass


if __name__ == "__main__":
    # main()
    y_true_metric()