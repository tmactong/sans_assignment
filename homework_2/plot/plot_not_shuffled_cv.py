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
        label='best n_estimators=100'
    )

    plt.xlabel("n_estimators")
    plt.ylabel(r"$R^2$")
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
        label='best max_features=5'
    )
    plt.xlabel("max_features")
    plt.ylabel(r"$R^2$")
    plt.ylim(0.67, 0.81)
    #plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_rf_max_features.png")


# ------------- Combined RF CV n_estimators and max_features -------------
def plot_rf_cv_n_estimators_and_max_features_combined():
    # ---------- data for n_estimators ----------
    n_estimators_list = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 250, 300, 400, 500, 600
    ]

    r2_list = [
        0.7899, 0.8003, 0.8150, 0.8082, 0.8156, 0.8033, 0.8113, 0.8154,
        0.8143, 0.8221, 0.8165, 0.8115, 0.8056, 0.8079, 0.8124, 0.8131
    ]

    best_idx = int(np.argmax(r2_list))
    best_n = n_estimators_list[best_idx]

    # ---------- data for max_features ----------
    max_features = [1, 2, 3, 4, 5, 6]
    r2_100 = [0.6755, 0.7174, 0.7553, 0.7906, 0.7993, 0.7974]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- subplot 1: n_estimators ----------
    ax = axes[0]
    ax.plot(n_estimators_list, r2_list, marker='o', color='tab:blue')
    ax.vlines(
        best_n, min(r2_list), max(r2_list),
        color='black', linestyles='dashed',
        label=f'best n_estimators={best_n}'
    )
    ax.set_xlabel("n_estimators")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.78, 0.83)
    ax.legend()

    # ---------- subplot 2: max_features ----------
    ax = axes[1]
    ax.plot(max_features, r2_100, marker='o',
            color='tab:blue', label="n_estimators=100")
    ax.vlines(
        5, min(r2_100), max(r2_100),
        color='black', linestyles='dashed',
        label='best max_features=5'
    )
    ax.set_xlabel("max_features")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.67, 0.81)
    ax.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("graphs/not_shuffled_cv_rf_n_and_max_features.png",bbox_inches="tight")

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
    plt.ylabel(r"$R^2$")
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
    plt.ylim(0.75, 0.82)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/not_shuffled_cv_rf_min_samples_split.png")

# ------------- Combined RF CV max_depth and min_samples_split -------------
def plot_rf_cv_max_depth_and_min_samples_split_combined():
    # ---------- data for max_depth ----------
    max_depth_list = [5, 10, 15, 20, 25, 30, 35, 40]

    r2_100_5 = [0.7246, 0.7856, 0.8036, 0.8082, 0.7993, 0.7954, 0.7891, 0.8003]
    r2_100_6 = [0.7284, 0.8202, 0.8129, 0.8055, 0.8102, 0.8150, 0.8147, 0.8146]

    # ---------- data for min_samples_split ----------
    min_samples_split_list = list(range(2, 16))

    r2_100_none = [
        0.8164, 0.8061, 0.8149, 0.8094, 0.8039, 0.8010, 0.7925, 0.7964,
        0.8011, 0.7822, 0.7835, 0.7789, 0.7770, 0.7678
    ]

    r2_100_10 = [
        0.8087, 0.8043, 0.8063, 0.7985, 0.7989, 0.8052, 0.8003, 0.7907,
        0.7905, 0.7832, 0.7894, 0.7742, 0.7596, 0.7671
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- subplot 1: max_depth ----------
    ax = axes[0]
    ax.plot(max_depth_list, r2_100_5, marker="o",
            color="tab:blue",
            label="n_estimators=100, n_features=5")

    ax.plot(max_depth_list, r2_100_6, marker="o",
            color="tab:green",
            label="n_estimators=100, n_features=6")

    ax.vlines(
        10, min(r2_100_6), max(r2_100_6),
        color="black", linestyles="dashed",
        label="best max_depth=10"
    )

    ax.set_xlabel("max_depth")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.72, 0.83)
    ax.legend()

    # ---------- subplot 2: min_samples_split ----------
    ax = axes[1]

    ax.plot(min_samples_split_list, r2_100_none,
            marker="o", color="tab:blue",
            label="n_estimators=100, n_features=6, max_depth=None")

    ax.plot(min_samples_split_list, r2_100_10,
            marker="o", color="tab:green",
            label="n_estimators=100, n_features=6, max_depth=10")

    ax.vlines(
        2, min(r2_100_none), max(r2_100_none),
        color="black", linestyles="dashed",
        label="best min_samples_split=2"
    )

    ax.set_xlabel("min_samples_split")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.75, 0.82)
    ax.set_xticks(min_samples_split_list)
    ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig("graphs/not_shuffled_cv_rf_max_depth_and_min_samples_split.png",bbox_inches="tight")

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
    # plot_rf_cv_n_estimators()
    # plot_rf_cv_max_features()
    # plot_rf_cv_max_depth()
    # plot_rf_cv_min_samples_split()
    # plot_rf_cv_n_estimators_and_max_features_combined()
    # plot_rf_cv_max_depth_and_min_samples_split_combined()
    # plot_knn_cv_k_neighbors()
    # plot_rf_cv_n_estimators_and_max_features_combined()
    # plot_rf_cv_max_depth_and_min_samples_split_combined()
    plot_rf_cv()