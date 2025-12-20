import matplotlib.pyplot as plt
import numpy as np

def plot_knn_k():
    """Matrix columns = 103"""
    # rank
    k = np.arange(1, 25)

    # RMSE
    rmse_est = np.array([
        16.83, 12.914, 12.564, 12.15, 11.956, 11.931, 11.882, 11.904,
        11.958, 11.96, 11.96, 11.975, 12.015, 12.02, 12.021, 12.042,
        12.057, 12.084, 12.146, 12.219, 12.3, 12.325, 12.334, 12.372
    ])

    rmse_ref = np.array([
        16.654, 12.913, 12.624, 12.21, 11.871, 11.887, 11.881, 11.926,
        11.933, 11.924, 11.892, 11.933, 11.952, 11.993, 11.994, 12.02,
        12.055, 12.105, 12.128, 12.23, 12.267, 12.304, 12.321, 12.372
    ])

    singular_values_ref = np.array([
        570.776746225811, 419.43479320286235, 261.5723218019054, 236.11470090466378, 177.08214613824535,
        145.873982422326, 141.11400552069088, 128.46136830117626, 109.90752325859842, 103.79688786967839,
        85.70837086425873, 73.72368346050192, 71.40913136400384, 67.3034328482223, 59.88019779766087,
        54.448879660237886, 47.23188167070224, 46.52732493720707, 40.139505325378124, 39.27710369185448,
        37.823804518177546, 35.81152151787856, 30.076360838374764, 22.198124213405652
    ])

    singular_values_est = np.array([
        528.6400325962268, 377.1353612079643, 241.6680783489871, 212.8352026836788, 165.5693582565852,
        142.43610922877036, 132.21085867222703, 118.3062694476132, 111.31020983830753, 97.14690076263744,
        86.0188321499834, 81.84356854833781, 74.04669232373259, 73.17525627915155, 69.00646899349817,
        63.686753767464964, 60.52951711141678, 57.61715315826657, 55.41217944907795, 47.40325215905876,
        46.150707119487976, 42.76618362593329, 40.120106119562266, 35.01546341372077
    ])

    best_idx_est = np.argmin(rmse_est)
    best_idx_ref = np.argmin(rmse_ref)
    best_k_est = k[best_idx_est]
    best_k_ref = k[best_idx_ref]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_ylim(11.5,17)
    axes[0].plot(k, rmse_est, marker='o', color='tab:blue')
    axes[0].plot(k, rmse_ref, marker='o', color='tab:green')

    axes[0].vlines(
        best_k_est, min(rmse_est),max(rmse_est),
        color='darkgreen', linestyles='dashed',
        label=f'Matrix X from Est. Data, best k={best_k_est}'
    )
    axes[0].vlines(
        best_k_ref, min(rmse_est), max(rmse_est),
        color='darkorange', linestyles='dashed',
        label=f'Matrix X from Ref. Data, best k={best_k_ref}'
    )

    axes[0].set_xlabel("Rank k")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].set_title("KNN denoising (RMSE)")

    axes[1].plot(k, singular_values_est, marker='o', label='Matrix X from Est. Data')
    axes[1].plot(k, singular_values_ref, marker='o', label='Matrix X from Ref. Data')
    axes[1].set_ylabel("Singular value")
    axes[1].set_title("Singular values of Matrix X")
    axes[1].legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig("graphs/knn_denoising_k=103.png")


def plot_rf_k():
    """Matrix columns = 92"""
    # rank
    k = np.arange(1, 25)

    # RMSE
    rmse_est = np.array([
        16.055, 12.5, 11.781, 10.876, 10.581, 10.315, 10.162, 10.034, 9.825, 9.738, 9.744, 9.661, 9.665, 9.635, 9.614,
        9.609, 9.631, 9.648, 9.63, 9.649, 9.653, 9.639, 9.663, 9.672
    ])

    rmse_ref = np.array([
        15.727, 12.502, 11.948, 10.991, 10.505, 10.289, 10.142, 10.006, 9.874, 9.759, 9.662, 9.628, 9.64, 9.635, 9.663,
        9.637, 9.647, 9.659, 9.639, 9.634, 9.637, 9.629, 9.639, 9.672
    ])

    singular_values_ref = np.array([
        542.3200558874292, 408.73853466994984, 248.2218672084431, 219.18479278572525, 172.18139305918808,
        141.51259877691052, 134.3491688690054, 121.9280277887817, 100.05645343636799, 95.3135238841573,
        80.00740169006286, 70.82411609253552, 67.25746328162973, 65.82694776579655, 57.87655314257236,
        50.96260267225961, 46.0350904940879, 44.91325488160035, 38.72942340569298, 37.95265562284009, 34.81959168386021,
        33.48339674381238, 27.59315929442263, 21.000214584988957
    ])

    singular_values_est = np.array([
        470.72408292346324, 337.9174978404642, 220.7975408250492, 187.03696493193027, 152.02941592078506,
        129.22924102987298, 119.56234455612427, 100.69579635022744, 96.41939486461284, 81.59279284948983,
        71.99667071525418, 66.12094047803214, 64.47237411816704, 63.18338938435673, 56.31909814955868, 52.2984614480492,
        49.72790593957304, 46.644394016854186, 41.52704995363443, 39.445737828815396, 35.83250769950375,
        32.57177696117058, 30.28995884845081, 25.18878047634629
    ])

    best_idx_est = np.argmin(rmse_est)
    best_idx_ref = np.argmin(rmse_ref)
    best_k_est = k[best_idx_est]
    best_k_ref = k[best_idx_ref]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_ylim(9,17)
    axes[0].plot(k, rmse_est, marker='o', color='tab:blue')
    axes[0].plot(k, rmse_ref, marker='o', color='tab:green')

    axes[0].vlines(
        best_k_est, min(rmse_est),max(rmse_est),
        color='darkgreen', linestyles='dashed',
        label=f'Matrix X from Est. Data, best k={best_k_est}'
    )
    axes[0].vlines(
        best_k_ref, min(rmse_est), max(rmse_est),
        color='darkorange', linestyles='dashed',
        label=f'Matrix X from Ref. Data, best k={best_k_ref}'
    )

    axes[0].set_xlabel("Rank k")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].set_title("Random Forest denoising (RMSE)")

    axes[1].plot(k, singular_values_est, marker='o', label='Matrix X from Est. Data')
    axes[1].plot(k, singular_values_ref, marker='o', label='Matrix X from Ref. Data')
    axes[1].set_ylabel("Singular value")
    axes[1].set_title("Singular values of Matrix X")
    axes[1].legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("graphs/rf_denoising_k=92.png")

if __name__ == '__main__':
    # plot_knn_k()
    plot_rf_k()