import matplotlib.pyplot as plt
import numpy as np

def plot_rf_cv_n_estimators():
    n_estimators_list = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
    ]

    r2_list = [
        0.8346, 0.8392, 0.8452, 0.8536, 0.8468, 0.8477, 0.8509, 0.8509, 0.8485, 0.8471, 0.8533, 0.8538, 0.8534, 0.8510,
        0.8528, 0.8553, 0.8530, 0.8514, 0.8520, 0.8516, 0.8525, 0.8527, 0.8543, 0.8521, 0.8503, 0.8484, 0.8509, 0.8503,
        0.8507, 0.8552
    ]

    best_idx = np.argmax(r2_list)
    best_n = n_estimators_list[best_idx]

    plt.figure(figsize=(10, 5))
    plt.ylim(0.83,0.86)

    plt.plot(n_estimators_list, r2_list, marker='o', color='tab:blue')

    plt.vlines(
        best_n, min(r2_list),max(r2_list),
        color='black', linestyles='dashed',
        label='best n_estimators=600'
    )

    plt.xlabel("n_estimators")
    plt.ylabel("R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/cv_rf_n_estimators.png")


def plot_rf_cv_max_features():
    max_features = [1, 2, 3, 4, 5, 6]

    r2_600 = [0.7097,0.7735,0.8095,0.8328,0.8424,0.8491]

    colors = {"n=600": "tab:blue"}

    plt.figure(figsize=(10, 5))

    # n_estimators=600
    plt.plot(max_features, r2_600, marker='o', color=colors["n=600"], label="n_estimators=600")

    plt.vlines(
        6, min(r2_600), max(r2_600),
        color='black', linestyles='dashed',
        label='best max_features=6'
    )
    plt.xlabel("max_features")
    plt.ylabel("R2")
    plt.title("Random Forest Cross-validation (max_features)")
    plt.ylim(0.70, 0.86)
    #plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/cv_rf_max_features.png")

def plot_rf_cv_n_and_max_features_combined():
    # ---- data for n_estimators ----
    n_estimators_list = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
    ]

    r2_list = [
        0.8346, 0.8392, 0.8452, 0.8536, 0.8468, 0.8477, 0.8509, 0.8509, 0.8485, 0.8471,
        0.8533, 0.8538, 0.8534, 0.8510, 0.8528, 0.8553, 0.8530, 0.8514, 0.8520, 0.8516,
        0.8525, 0.8527, 0.8543, 0.8521, 0.8503, 0.8484, 0.8509, 0.8503, 0.8507, 0.8552
    ]

    best_idx = int(np.argmax(r2_list))
    best_n = n_estimators_list[best_idx]

    # ---- data for max_features ----
    max_features = [1, 2, 3, 4, 5, 6]
    r2_600 = [0.7097, 0.7735, 0.8095, 0.8328, 0.8424, 0.8491]

    # ---- create figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- subplot 1: n_estimators ----
    ax = axes[0]
    ax.plot(n_estimators_list, r2_list, marker='o', color='tab:blue')
    ax.vlines(best_n, min(r2_list), max(r2_list),
              color='black', linestyles='dashed',
              label=f'best n_estimators={best_n}')
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("R2")
    ax.set_ylim(0.83, 0.86)
    ax.legend()

    # ---- subplot 2: max_features ----
    ax = axes[1]
    ax.plot(max_features, r2_600, marker='o',
            color='tab:blue', label="n_estimators=600")
    ax.vlines(6, min(r2_600), max(r2_600),
              color='black', linestyles='dashed',
              label='best max_features=6')
    ax.set_xlabel("max_features")
    ax.set_ylabel("R2")
    ax.set_ylim(0.70, 0.86)
    ax.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("graphs/cv-shuffled-rf-n_and_max_features.png",
                bbox_inches="tight")

def plot_rf_cv_max_depth():
    max_depth_list = [5, 10, 15, 20, 25, 30, 35,40,45,50]

    r2_600 = [0.7299,0.8435,0.8532,0.8521,0.8560,0.8523,0.8491,0.8554,0.8511,0.8498]

    plt.figure(figsize=(10, 5))

    colors = {
        "n=600": "tab:blue"
    }

    # n_estimators = 600
    plt.plot(max_depth_list, r2_600, marker="o", color=colors["n=600"], label="n_estimators=600,n_features=6")


    plt.vlines(
        25, min(r2_600), max(r2_600),
        color='black', linestyles='dashed',
        label='best max_depth=25'
    )

    plt.xlabel("max_depth")
    plt.ylabel(r"$R^2$")
    plt.ylim(0.72, 0.86)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/cv_rf_max_depth.png")

def plot_rf_cv_min_samples_split():
    min_samples_split_list = list(range(2, 16))  # 2~15

    # n_estimators=600, max_depth=None
    r2_600_none = [
        0.8496, 0.8511, 0.8522, 0.8527, 0.8500, 0.8432, 0.8457, 0.8473, 0.8474, 0.8411, 0.8419, 0.8375, 0.8398, 0.8323
    ]

    # n_estimators=600, max_depth=25
    r2_600_25 = [
        0.8529, 0.8519, 0.8496, 0.8497, 0.8470, 0.8450, 0.8502, 0.8467, 0.8435, 0.8424, 0.8439, 0.8365, 0.8369, 0.8342
    ]

    plt.figure(figsize=(10, 6))

    def plot_one(r2_list, label, color):
        best_idx = int(np.argmax(r2_list))

        plt.plot(min_samples_split_list, r2_list,
                 marker="o", label=label, color=color)

    plot_one(r2_600_none, "n_estimators=600,n_features=6,max_depth=None", "tab:blue")
    plot_one(r2_600_25,   "n_estimators=600,n_features=6,max_depth=25",   "tab:green")


    plt.vlines(
        2, min(r2_600_none), max(r2_600_none),
        color='black', linestyles='dashed',
        label='min_samples_split=2'
    )
    plt.xlabel("min_samples_split")
    plt.xticks([2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15])
    plt.ylabel(r"$R^2$")
    plt.ylim(0.83, 0.86)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/cv_rf_min_samples_split.png")


# --- Combined max_depth and min_samples_split plot ---
def plot_rf_cv_max_depth_and_min_samples_split_combined():
    # ---------- data for max_depth ----------
    max_depth_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    r2_600 = [0.7299, 0.8435, 0.8532, 0.8521, 0.8560,
              0.8523, 0.8491, 0.8554, 0.8511, 0.8498]

    # ---------- data for min_samples_split ----------
    min_samples_split_list = list(range(2, 16))

    r2_600_none = [
        0.8496, 0.8511, 0.8522, 0.8527, 0.8500, 0.8432,
        0.8457, 0.8473, 0.8474, 0.8411, 0.8419, 0.8375,
        0.8398, 0.8323
    ]

    r2_600_25 = [
        0.8529, 0.8519, 0.8496, 0.8497, 0.8470, 0.8450,
        0.8502, 0.8467, 0.8435, 0.8424, 0.8439, 0.8365,
        0.8369, 0.8342
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- subplot 1: max_depth ----------
    ax = axes[0]
    ax.plot(max_depth_list, r2_600, marker="o",
            color="tab:blue",
            label="n_estimators=600, n_features=6")

    ax.vlines(
        25, min(r2_600), max(r2_600),
        color="black", linestyles="dashed",
        label="best max_depth=25"
    )

    ax.set_xlabel("max_depth")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.72, 0.86)
    ax.legend()

    # ---------- subplot 2: min_samples_split ----------
    ax = axes[1]

    ax.plot(min_samples_split_list, r2_600_none,
            marker="o", color="tab:blue",
            label="n_estimators=600, n_features=6, max_depth=None")

    ax.plot(min_samples_split_list, r2_600_25,
            marker="o", color="tab:green",
            label="n_estimators=600, n_features=6, max_depth=25")

    ax.vlines(
        2, min(r2_600_none), max(r2_600_none),
        color="black", linestyles="dashed",
        label="best min_samples_split=2"
    )

    ax.set_xlabel("min_samples_split")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.81, 0.86)
    ax.set_xticks(min_samples_split_list)
    ax.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("graphs/cv_rf_max_depth_and_min_samples_split.png", bbox_inches="tight")


def plot_rf_cv():
    n_estimators_list = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500, 1600
    ]

    # max_features=6, max_depth=30, min_samples_split=2
    r2_list = [
        0.8221, 0.8291, 0.8330, 0.8342, 0.8359, 0.8366, 0.8374, 0.8379, 0.8379, 0.8385, 0.8399, 0.8405, 0.8405, 0.8404,
        0.8408, 0.8407, 0.8409, 0.8407, 0.8406, 0.8406, 0.8404, 0.8404, 0.8406, 0.8404, 0.8403
    ]

    # max_features=5, max_depth=30, min_samples_split=2
    r2_list_1 = [
        0.8107, 0.8198, 0.8228, 0.8248, 0.8263, 0.8265, 0.8277, 0.8280, 0.8286, 0.8294, 0.8300, 0.8315, 0.8318, 0.8318,
        0.8326, 0.8324, 0.8327, 0.8323, 0.8322, 0.8319, 0.8320, 0.8321, 0.8322, 0.8322, 0.8322
    ]

    # max_features=6, max_depth=30, min_samples_split=3
    r2_list_2 = [
        0.8193, 0.8272, 0.8317, 0.8324, 0.8346, 0.8357, 0.8360, 0.8371, 0.8373, 0.8380, 0.8394, 0.8400, 0.8400, 0.8400,
        0.8403, 0.8403, 0.8405, 0.8402, 0.8401, 0.8401, 0.8401, 0.8401, 0.8402, 0.8400, 0.8400,
    ]

    # max_features=6, max_depth=20, min_samples_split=2

    r2_list_3 = [
        0.8225, 0.8292, 0.8331, 0.8343, 0.8360, 0.8367, 0.8375, 0.8380, 0.8379, 0.8385, 0.8398, 0.8405, 0.8404, 0.8404,
        0.8408, 0.8407, 0.8409, 0.8407, 0.8406, 0.8406, 0.8404, 0.8404, 0.8406, 0.8404, 0.8402
    ]


    plt.figure(figsize=(10, 5))
    plt.ylim(0.805, 0.845)

    plt.plot(n_estimators_list, r2_list, marker='o', color='tab:blue',label="max_features=6,max_depth=30,min_samples_split=2")
    plt.plot(n_estimators_list, r2_list_1, marker='o', color='tab:orange', label="max_features=5,max_depth=30,min_samples_split=2")
    plt.plot(n_estimators_list, r2_list_2, marker='o', color='tab:green', label="max_features=6,max_depth=30,min_samples_split=3")
    plt.plot(n_estimators_list, r2_list_3, marker='o', color='tab:red',label="max_features=6,max_depth=20,min_samples_split=2")

    plt.vlines(
        800, min(r2_list_1), max(r2_list),
        color='black', linestyles='dashed',
        label='n_estimators=800, n_features=6, max_depth=30,min_samples_split=2'
    )

    plt.xlabel("n_estimators")
    plt.ylabel(r"$R^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/cv_rf.png")

def plot_knn_cv_k_neighbors():
    k_list = list(range(1, 16))

    # weights = 'uniform'
    r2_uniform = [
        0.7092, 0.7772, 0.7922, 0.7936, 0.7959, 0.7958, 0.7954, 0.7901, 0.7883, 0.7824, 0.7783, 0.7757, 0.7732, 0.7710,
        0.7684
    ]

    # weights = 'distance'
    r2_distance = [
        0.7092, 0.7814, 0.7981, 0.8021, 0.8051, 0.8060, 0.8061, 0.8023, 0.8011, 0.7967, 0.7934, 0.7911, 0.7888, 0.7870,
        0.7847
    ]

    plt.figure(figsize=(10, 5))
    plt.ylim(0.7, 0.81)
    plt.xlim(0,16)
    plt.xticks(k_list)

    # ---- uniform ----

    plt.plot(k_list, r2_uniform, marker='o',
             label="weights=uniform", color='tab:blue')

    # ---- distance ----

    plt.plot(k_list, r2_distance, marker='o',
             label="weights=distance", color='tab:orange')
    plt.vlines(
        7, min(r2_distance), max(r2_distance),
        color='black', linestyles='dashed',
        label='k_neighbors=7, weights=distance'
    )

    plt.xlabel("k_neighbors")
    plt.ylabel(r"$R^2$")
    # plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("graphs/cv_knn_k_neighbors.png")
    #plt.show()


if __name__ == '__main__':
    #plot_rf_cv_n_estimators()
    #plot_rf_cv_max_features()
    # plot_rf_cv_n_and_max_features_combined()
    # plot_rf_cv_max_depth_and_min_samples_split_combined()
    # plot_rf_cv_max_depth()
    # plot_rf_cv_min_samples_split()
    # plot_knn_cv_k_neighbors()
    plot_rf_cv()
    # plot_rf_cv_combined()
    # plot_rf_cv_max_depth_and_min_samples_split_combined()