import matplotlib.pyplot as plt
import numpy as np


def plot_knn_cv_k_neighbors():
    k_list = list(range(1, 16))

    # weights = 'uniform'
    r2_uniform = [
        0.7438, 0.7796, 0.7830, 0.7881, 0.7823, 0.7685, 0.7588, 0.7600, 0.7536, 0.7474, 0.7427, 0.7402, 0.7315, 0.7275,
        0.7224
    ]

    # weights = 'distance'
    r2_distance = [
        0.7438, 0.7856, 0.7935, 0.8014, 0.7976, 0.7873, 0.7799, 0.7805, 0.7751, 0.7701, 0.7660, 0.7638, 0.7570, 0.7529,
        0.7487
    ]

    plt.figure(figsize=(10, 5))
    plt.ylim(0.72, 0.81)
    plt.xlim(0,16)
    plt.xticks(k_list)

    # ---- uniform ----

    plt.plot(k_list, r2_uniform, marker='o',
             label="weights=uniform", color='tab:blue')

    # ---- distance ----

    plt.plot(k_list, r2_distance, marker='o',
             label="weights=distance", color='tab:orange')
    plt.vlines(
        4, min(r2_distance), max(r2_distance),
        color='black', linestyles='dashed',
        label='k_neighbors=4,weights=distance'
    )

    plt.xlabel("k_neighbors")
    plt.ylabel(r"$R^2$")
    # plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_knn_k_neighbors.png")
    #plt.show()

def plot_rf_cv():
    n_estimators_list = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500
    ]

    # max_features=6,max_depth=15,min_samples_split=2
    r2_list = [
        0.7594, 0.7727, 0.7747, 0.7769, 0.7775, 0.7779, 0.7767, 0.7764, 0.7765, 0.7779, 0.7824, 0.7811, 0.7809, 0.7802
    ]

    # max_features=5,max_depth=15,min_samples_split=2
    r2_list_1 = [
        0.7352, 0.7579, 0.7631, 0.7619, 0.7640, 0.7649, 0.7671, 0.7662, 0.7676, 0.7696, 0.7726, 0.7712, 0.7715, 0.7720
    ]

    # max_features=6,max_depth=10,min_samples_split=2
    r2_list_2 = [
        0.7602, 0.7731, 0.7731, 0.7736, 0.7733, 0.7735, 0.7729, 0.7714, 0.7715, 0.7730, 0.7786, 0.7779, 0.7782, 0.7780
    ]

    # max_features=6,max_depth=15,min_samples_split=3
    r2_list_3 = [
        0.7657, 0.7756, 0.7765, 0.7757, 0.7762, 0.7758, 0.7750, 0.7736, 0.7731, 0.7744, 0.7810, 0.7796, 0.7798, 0.7798
    ]

    plt.figure(figsize=(10, 5))
    plt.ylim(0.7,0.8)

    plt.plot(n_estimators_list, r2_list, marker='o', color='tab:blue',label="max_features=6,max_depth=15,min_samples_split=2")
    plt.plot(n_estimators_list, r2_list_1, marker='o', color='tab:orange',label="max_features=5,max_depth=15,min_samples_split=2")
    plt.plot(n_estimators_list, r2_list_2, marker='o', color='tab:red',
             label="max_features=6,max_depth=10,min_samples_split=2")
    plt.plot(n_estimators_list, r2_list_3, marker='o', color='tab:purple',
             label="max_features=6,max_depth=15,min_samples_split=3")

    plt.vlines(
        200, min(r2_list_1),max(r2_list),
        color='black', linestyles='dashed',
        label='n_estimators=200,max_features=6,max_depth=15,min_samples_split=2'
    )

    plt.xlabel("n_estimators")
    plt.ylabel(r"$R^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_rf.png")


if __name__ == '__main__':
    # plot_knn_cv_k_neighbors()
    plot_rf_cv()