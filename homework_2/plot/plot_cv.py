import matplotlib.pyplot as plt


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
    # plot_knn_cv_k_neighbors()
    plot_rf_cv()