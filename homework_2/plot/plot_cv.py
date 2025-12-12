import matplotlib.pyplot as plt
import numpy as np

def plot_rf_cv_n_estimators():
    n_estimators_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                         200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]

    r2_list = [0.7268, 0.7391, 0.7439, 0.7427, 0.7436, 0.7425, 0.7416, 0.7411, 0.7404, 0.7404,
               0.7406, 0.7398, 0.7403, 0.7403, 0.7409, 0.7408, 0.7407, 0.7408, 0.7407, 0.7406]

    best_idx = np.argmax(r2_list)
    best_n = n_estimators_list[best_idx]

    plt.figure(figsize=(10, 5))
    plt.ylim(0.725,0.748)

    plt.plot(n_estimators_list, r2_list, marker='o', color='tab:blue')

    plt.vlines(
        best_n, min(r2_list),max(r2_list),
        color='black', linestyles='dashed',
        label='n_estimators=30'
    )

    plt.xlabel("n_estimators")
    plt.ylabel("R2")
    plt.title("Random Forest Cross-validation (n_estimators)")
    plt.legend()
    plt.savefig("graphs/cv_rf_n_estimators.png",bbox_inches="tight")


def plot_rf_cv_max_features():
    max_features = [1, 2, 3, 4, 5, 6]

    r2_30 = [0.5181, 0.6136, 0.6682, 0.7134, 0.7290, 0.7439]
    r2_50 = [0.5331, 0.6197, 0.6744, 0.7090, 0.7302, 0.7436]
    r2_100 = [0.5280, 0.6185, 0.6714, 0.7091, 0.7283, 0.7404]

    colors = {
        "n=30": "tab:blue",
        "n=50": "tab:green",
        "n=100": "tab:red"
    }

    plt.figure(figsize=(10, 6))

    # n_estimators=30
    plt.plot(max_features, r2_30, marker='o', color=colors["n=30"], label="n_estimators=30")

    # n_estimators=50
    plt.plot(max_features, r2_50, marker='o', color=colors["n=50"], label="n_estimators=50")

    # n_estimators=100
    plt.plot(max_features, r2_100, marker='o', color=colors["n=100"], label="n_estimators=100")

    plt.vlines(
        6, min(r2_30), max(r2_30),
        color='black', linestyles='dashed',
        label='max_features=6'
    )
    plt.xlabel("max_features")
    plt.ylabel("R2")
    plt.title("Random Forest Cross-validation (max_features)")
    plt.ylim(0.50, 0.76)
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/cv_rf_max_features.png",bbox_inches="tight")

def plot_rf_cv_max_depth():
    max_depth_list = [5, 10, 15, 20, 25, 30, 35]

    r2_30 = [0.6038, 0.7392, 0.7419, 0.7444, 0.7439, 0.7439, 0.7439]
    r2_50 = [0.6024, 0.7381, 0.7426, 0.7436, 0.7436, 0.7436, 0.7436]
    r2_100 = [0.5950, 0.7354, 0.7413, 0.7411, 0.7405, 0.7404, 0.7404]

    plt.figure(figsize=(10, 6))

    colors = {
        "n=30": "tab:blue",
        "n=50": "tab:green",
        "n=100": "tab:red"
    }

    # n_estimators = 30
    plt.plot(max_depth_list, r2_30, marker="o", color=colors["n=30"], label="n_estimators=30,n_features=6")

    # --- n_estimators = 50 ---
    plt.plot(max_depth_list, r2_50, marker="o", color=colors["n=50"], label="n_estimators=50,n_features=6")

    # --- n_estimators = 100 ---
    plt.plot(max_depth_list, r2_100, marker="o", color=colors["n=100"], label="n_estimators=100,n_features=6")

    plt.vlines(
        20, min(r2_30), max(r2_30),
        color='black', linestyles='dashed',
        label='max_depth=20'
    )

    plt.xlabel("max_depth")
    plt.ylabel("R2")
    plt.title("Random Forest Cross-validation (max_depth)")
    plt.ylim(0.58, 0.76)
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/cv_rf_max_depth.png",bbox_inches="tight")

def plot_rf_cv_min_samples_split():
    min_samples_split_list = list(range(2, 16))  # 2~15

    # n_estimators=30, max_depth=None
    r2_30_none = [
        0.7439, 0.7441, 0.7453, 0.7446, 0.7452,
        0.7437, 0.7434, 0.7400, 0.7400, 0.7385,
        0.7363, 0.7339, 0.7321, 0.7296
    ]

    # n_estimators=30, max_depth=20
    r2_30_20 = [
        0.7444, 0.7442, 0.7452, 0.7445, 0.7453,
        0.7437, 0.7433, 0.7400, 0.7400, 0.7385,
        0.7363, 0.7339, 0.7321, 0.7296
    ]

    # n_estimators=100, max_depth=None
    r2_50_none = [
        0.7436, 0.7429, 0.7445, 0.7439, 0.7439,
        0.7428, 0.7422, 0.7398, 0.7398, 0.7383,
        0.7365, 0.7343, 0.7325, 0.7312
    ]

    r2_50_20 = [
        0.7436, 0.7429, 0.7443, 0.7437, 0.7439,
        0.7428, 0.7422, 0.7398, 0.7398, 0.7383,
        0.7365, 0.7343, 0.7325, 0.7312
    ]

    # n_estimators=100, max_depth=None
    r2_100_none = [
        0.7404, 0.7402, 0.7423, 0.7418, 0.7413,
        0.7408, 0.7400, 0.7383, 0.7380, 0.7362,
        0.7341, 0.7323, 0.7305, 0.7283
    ]

    # n_estimators=100, max_depth=20
    r2_100_20 = [
        0.7411, 0.7403, 0.7424, 0.7418, 0.7414,
        0.7408, 0.7399, 0.7383, 0.7380, 0.7362,
        0.7341, 0.7323, 0.7305, 0.7283
    ]

    plt.figure(figsize=(10, 6))

    def plot_one(r2_list, label, color):
        best_idx = int(np.argmax(r2_list))

        plt.plot(min_samples_split_list, r2_list,
                 marker="o", label=label, color=color)

    plot_one(r2_30_none, "n_estimators=30,n_features=6,max_depth=None", "tab:blue")
    plot_one(r2_30_20,   "n_estimators=30,n_features=6,max_depth=20",   "tab:green")
    plot_one(r2_100_none,"n_estimators=100,n_features=6,max_depth=None","tab:red")
    plot_one(r2_100_20,  "n_estimators=100,n_features=6,max_depth=20",  "tab:purple")
    plot_one(r2_50_none, "n_estimators=50,n_features=6,max_depth=None", "tab:cyan")
    plot_one(r2_50_20, "n_estimators=50,n_features=6,max_depth=20", "tab:olive")

    plt.vlines(
        4, min(r2_30_none), max(r2_30_none),
        color='black', linestyles='dashed',
        label='min_samples_split=4'
    )
    plt.xlabel("min_samples_split")
    plt.ylabel("R2")
    plt.title("Random Forest Cross-Validation (min_samples_split)")
    plt.ylim(0.728, 0.747)
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/cv_rf_min_samples_split.png", bbox_inches="tight")

def plot_knn_cv_k_neighbors():
    k_list = list(range(1, 16))

    # weights = 'uniform'
    r2_uniform = [
        0.5489, 0.6480, 0.6636, 0.6759, 0.6821,
        0.6833, 0.6824, 0.6817, 0.6827, 0.6821,
        0.6809, 0.6808, 0.6777, 0.6751, 0.6718
    ]

    # weights = 'distance'
    r2_distance = [
        0.5489, 0.6492, 0.6670, 0.6795, 0.6859,
        0.6878, 0.6873, 0.6870, 0.6883, 0.6880,
        0.6874, 0.6874, 0.6851, 0.6828, 0.6800
    ]

    plt.figure(figsize=(10, 6))
    plt.ylim(0.54, 0.70)

    # ---- uniform ----

    plt.plot(k_list, r2_uniform, marker='o',
             label="weights=uniform", color='tab:blue')

    # ---- distance ----

    plt.plot(k_list, r2_distance, marker='o',
             label="weights=distance", color='tab:orange')
    plt.vlines(
        6, min(r2_distance), max(r2_distance),
        color='black', linestyles='dashed',
        label='k_neighbors=6'
    )

    plt.xlabel("k_neighbors")
    plt.ylabel("R2")
    plt.title("KNN Cross-validation (k_neighbors)")
    plt.grid(True)
    plt.legend()

    plt.savefig("graphs/cv_knn_k_neighbors.png", bbox_inches="tight")
    #plt.show()


if __name__ == '__main__':
    # plot_rf_cv_n_estimators()
    # plot_rf_cv_max_features()
    # plot_rf_cv_max_depth()
    # plot_rf_cv_min_samples_split()
    plot_knn_cv_k_neighbors()