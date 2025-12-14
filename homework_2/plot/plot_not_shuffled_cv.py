import matplotlib.pyplot as plt
import numpy as np

def plot_rf_cv_n_estimators():
    n_estimators_list = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 250, 300, 400, 500, 600
    ]

    r2_list = [
        0.7899, 0.8003, 0.8150, 0.8082, 0.8156, 0.8033, 0.8113, 0.8154, 0.8143, 0.8221, 0.8165, 0.8115, 0.8056, 0.8079,
        0.8124, 0.8131
    ]

    best_idx = np.argmax(r2_list)
    best_n = n_estimators_list[best_idx]

    plt.figure(figsize=(10, 5))
    plt.ylim(0.78,0.83)

    plt.plot(n_estimators_list, r2_list, marker='o', color='tab:blue')

    plt.vlines(
        best_n, min(r2_list),max(r2_list),
        color='black', linestyles='dashed',
        label='n_estimators=100'
    )

    plt.xlabel("n_estimators")
    plt.ylabel(r"$R^2$")
    plt.title("Random Forest Cross-validation (n_estimators)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_rf_n_estimators.png")


def plot_rf_cv_max_features():
    max_features = [1, 2, 3, 4, 5, 6]

    r2_100 = [0.6755,0.7174,0.7553,0.7906,0.7993,0.7974]

    colors = {"n=100": "tab:blue"}

    plt.figure(figsize=(10, 5))

    # n_estimators=100
    plt.plot(max_features, r2_100, marker='o', color=colors["n=100"], label="n_estimators=100")

    plt.vlines(
        5, min(r2_100), max(r2_100),
        color='black', linestyles='dashed',
        label='max_features=5'
    )
    plt.xlabel("max_features")
    plt.ylabel("R2")
    plt.title("Random Forest Cross-validation (max_features)")
    plt.ylim(0.67, 0.81)
    #plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_rf_max_features.png")

def plot_rf_cv_max_depth():
    max_depth_list = [5, 10, 15, 20, 25, 30, 35,40]

    r2_100_5 = [0.7246,0.7856,0.8036,0.8082,0.7993,0.7954,0.7891,0.8003]
    r2_100_6 = [0.7284,0.8202,0.8129,0.8055,0.8102,0.8150,0.8147,0.8146]

    plt.figure(figsize=(10, 5))

    colors = {
        "n=5": "tab:blue",
        "n=6": "tab:green"
    }

    plt.plot(max_depth_list, r2_100_5, marker="o", color=colors["n=5"], label="n_estimators=100,n_features=5")
    plt.plot(max_depth_list, r2_100_6, marker="o", color=colors["n=6"], label="n_estimators=100,n_features=6")

    plt.vlines(
        10, min(r2_100_6), max(r2_100_6),
        color='black', linestyles='dashed',
        label='max_depth=10,n_features=6'
    )

    plt.xlabel("max_depth")
    plt.ylabel("R2")
    plt.title("Random Forest Cross-validation (max_depth)")
    plt.ylim(0.72, 0.83)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_rf_max_depth.png")

def plot_rf_cv_min_samples_split():
    min_samples_split_list = list(range(2, 16))  # 2~15

    # n_estimators=100, max_depth=None
    r2_100_none = [
        0.8164, 0.8061, 0.8149, 0.8094, 0.8039, 0.8010, 0.7925, 0.7964, 0.8011, 0.7822, 0.7835, 0.7789, 0.7770, 0.7678
    ]

    # n_estimators=100, max_depth=10
    r2_100_10 = [
        0.8087, 0.8043, 0.8063, 0.7985, 0.7989, 0.8052, 0.8003, 0.7907, 0.7905, 0.7832, 0.7894, 0.7742, 0.7596, 0.7671
    ]

    plt.figure(figsize=(10, 5))

    def plot_one(r2_list, label, color):
        best_idx = int(np.argmax(r2_list))

        plt.plot(min_samples_split_list, r2_list,
                 marker="o", label=label, color=color)

    plot_one(r2_100_none, "n_estimators=100,n_features=6,max_depth=None", "tab:blue")
    plot_one(r2_100_10,   "n_estimators=100,n_features=6,max_depth=10",   "tab:green")


    plt.vlines(
        2, min(r2_100_none), max(r2_100_none),
        color='black', linestyles='dashed',
        label='min_samples_split=2,max_depth=None'
    )
    plt.xlabel("min_samples_split")
    plt.xticks([2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15])
    plt.ylabel(r"$R^2$")
    plt.title("Random Forest Cross-Validation (min_samples_split)")
    plt.ylim(0.75, 0.82)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_rf_min_samples_split.png")

def plot_knn_cv_k_neighbors():
    k_list = list(range(1, 16))

    # weights = 'uniform'
    r2_uniform = [
        0.7422, 0.8029, 0.8052, 0.8069, 0.8038, 0.7985, 0.7963, 0.7873, 0.7878, 0.7873, 0.7772, 0.7683, 0.7669, 0.7623,
        0.7584
    ]

    # weights = 'distance'
    r2_distance = [
        0.7356, 0.7993, 0.8251, 0.8068, 0.8167, 0.8045, 0.8150, 0.8146, 0.8122, 0.8084, 0.8013, 0.7922, 0.7866, 0.7801,
        0.7839
    ]

    plt.figure(figsize=(10, 6))
    plt.ylim(0.73, 0.83)
    plt.xlim(0,16)
    plt.xticks(k_list)

    # ---- uniform ----

    plt.plot(k_list, r2_uniform, marker='o',
             label="weights=uniform", color='tab:blue')

    # ---- distance ----

    plt.plot(k_list, r2_distance, marker='o',
             label="weights=distance", color='tab:orange')
    plt.vlines(
        3, min(r2_distance), max(r2_distance),
        color='black', linestyles='dashed',
        label='k_neighbors=3,weights=distance'
    )

    plt.xlabel("k_neighbors")
    plt.ylabel("R2")
    plt.title("KNN Cross-validation (k_neighbors)")
    # plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_knn_k_neighbors.png")
    #plt.show()


if __name__ == '__main__':
    # plot_rf_cv_n_estimators()
    # plot_rf_cv_max_features()
    # plot_rf_cv_max_depth()
    plot_rf_cv_min_samples_split()
    # plot_knn_cv_k_neighbors()