import matplotlib.pyplot as plt
import numpy as np

def plot_knn_k():
    """Matrix columns = 105"""
    # rank
    k = np.arange(1, 25)

    # RMSE
    rmse_est = np.array([
        16.827, 12.811, 12.612, 12.251, 12.057, 12.041, 11.963, 12.004, 12.088, 12.114, 12.14, 12.065, 12.045, 12.054,
        12.065, 12.067, 12.073, 12.104, 12.147, 12.198, 12.236, 12.266, 12.295, 12.304
    ])

    rmse_ref = np.array([
        16.544, 12.807, 12.669, 12.292, 11.956, 11.996, 11.97, 12.038, 12.039, 12.022, 11.966, 11.998, 11.992, 12.009,
        12.015, 12.048, 12.069, 12.099, 12.104, 12.176, 12.198, 12.233, 12.261, 12.304
    ])

    singular_values_est = np.array([
        552.8691507186547, 386.91888251608833, 247.8310171561914, 212.29065528864223, 165.90067578360686,
        144.05804459667837, 132.24480327743404, 119.40376387351073, 112.55021348600496, 97.90268168851416,
        85.08217077539202, 83.3318227804199, 74.43503351403054, 73.83492212413128, 68.51212335646316, 64.43947364893744,
        63.47835575890353, 58.249203223271365, 56.09873043617007, 48.796962100826406, 45.21270813277913,
        42.59338061886889, 41.00608514338988, 38.7884082704928
    ])

    singular_values_ref = np.array([
        594.1988044045762, 436.0440125318794, 267.3392446489093, 238.04268797703995, 178.56296485715265,
        147.29341306850492, 141.85643203350506, 130.23601860760067, 110.027560676994, 103.87699507488891,
        85.79237944446054, 73.92890860236675, 71.65254528223454, 67.41953158762763, 60.028372285162504,
        55.40439871313331, 47.35080102451734, 46.75515762046923, 40.367281955534544, 39.46657213849704,
        37.98067646056571, 36.136124853202766, 30.402152400425642, 22.230621584483863
    ])

    best_idx_est = np.argmin(rmse_est)
    best_idx_ref = np.argmin(rmse_ref)
    best_k_est = k[best_idx_est]
    best_k_ref = k[best_idx_ref]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_ylim(11.5,17)
    axes[0].plot(k, rmse_est, marker='o', color='tab:green')
    axes[0].plot(k, rmse_ref, marker='o', color='tab:orange')

    axes[0].vlines(
        best_k_est, min(rmse_est),max(rmse_est),
        color='darkgreen', linestyles='dashed',
        label=f'Matrix X from Est. Data, optimal k={best_k_est}'
    )
    axes[0].vlines(
        best_k_ref, min(rmse_est), max(rmse_est),
        color='darkorange', linestyles='dashed',
        label=f'Matrix X from Ref. Data, optimal k={best_k_ref}'
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
    plt.savefig("graphs/knn_denoising_k=105.png")


def plot_rf_k():
    """Matrix columns = 93"""
    # rank
    k = np.arange(1, 25)

    # RMSE
    rmse_est = np.array([
        15.708, 12.182, 11.42, 10.579, 10.312, 10.035, 9.88, 9.684, 9.574, 9.494, 9.508, 9.424, 9.427, 9.404, 9.376,
        9.374, 9.4, 9.421, 9.397, 9.394, 9.399, 9.385, 9.407, 9.416
    ])

    rmse_ref = np.array([
        15.406, 12.195, 11.609, 10.673, 10.198, 9.98, 9.864, 9.723, 9.591, 9.504, 9.401, 9.384, 9.398, 9.389, 9.402,
        9.371, 9.391, 9.406, 9.389, 9.387, 9.38, 9.373, 9.383, 9.416
    ])

    singular_values_est = np.array([
        471.8885910257943, 343.37294020518783, 220.8237786903374, 191.60656830143822, 152.76920218341627,
        129.29872055300692, 120.87956887417845, 102.21437822882305, 99.68173434695012, 82.03809398996371,
        71.73480973538655, 67.93775757989518, 64.96354291992513, 63.640382071379754, 56.23203802794691,
        52.60927975149195, 49.98617116258176, 46.526088199928594, 41.61717053661345, 39.86783633242177,
        36.448121160979234, 34.282532800238144, 30.04877028194473, 25.431442133914558
    ])

    singular_values_ref = np.array([
        546.1198250019733, 412.77085166310445, 248.7990776853743, 222.5824950406764, 173.16309351820263,
        141.57576165811918, 135.63228758039915, 123.72997888297913, 100.91660970041896, 96.49018632736419,
        80.53277603997338, 72.01213057136434, 67.26129300757476, 66.11223025406441, 57.88416160936531,
        51.56966805538874, 46.732846328023676, 45.06560629045032, 38.94613933223935, 37.963203090046356,
        34.82988756240485, 34.019923620433346, 27.734168424308173, 21.00793592887436
    ])

    best_idx_est = np.argmin(rmse_est)
    best_idx_ref = np.argmin(rmse_ref)
    best_k_est = k[best_idx_est]
    best_k_ref = k[best_idx_ref]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_ylim(9,17)
    axes[0].plot(k, rmse_est, marker='o', color='tab:green')
    axes[0].plot(k, rmse_ref, marker='o', color='tab:orange')

    axes[0].vlines(
        best_k_est, min(rmse_est),max(rmse_est),
        color='darkgreen', linestyles='dashed',
        label=f'Matrix X from Est. Data, optimal k={best_k_est}'
    )
    axes[0].vlines(
        best_k_ref, min(rmse_est), max(rmse_est),
        color='darkorange', linestyles='dashed',
        label=f'Matrix X from Ref. Data, optimal k={best_k_ref}'
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
    plt.savefig("graphs/rf_denoising_k=93.png")

def plot_mlr_k():
    """Matrix columns = 100"""
    # rank
    k = np.arange(1, 25)

    # RMSE
    rmse_est = np.array([
        16.711, 13.175, 12.173, 11.352, 10.846, 10.55, 10.304, 10.046, 9.87, 9.817, 9.743, 9.766, 9.716, 9.65, 9.555,
        9.522, 9.468, 9.38, 9.349, 9.327, 9.31, 9.275, 9.279, 9.262
    ])

    rmse_ref = np.array([
        16.276, 12.936, 12.235, 11.1, 10.571, 10.214, 10.03, 9.849, 9.747, 9.659, 9.553, 9.554, 9.484, 9.506, 9.463,
        9.414, 9.391, 9.365, 9.338, 9.325, 9.3, 9.261, 9.242, 9.262
    ])

    singular_values_est = np.array([
        477.12079460346035, 415.47003397395986, 235.75031465337852, 223.84900378439542, 151.04423758158862,
        135.397919998688, 120.00052103485973, 112.99250869005797, 97.26670504582256, 87.5234099152215,
        85.05855259132468, 72.14115391277335, 65.74415591530428, 61.649405217774, 59.38839436151427, 52.83101981728861,
        52.38754954775994, 49.11030954253323, 46.78511037632876, 42.24305765855134, 40.430177281369275,
        38.89710962498675, 34.384715762605374, 26.305451843829594
    ])

    singular_values_ref = np.array([
        553.901777221338, 418.02824543795117, 251.09268033941254, 227.07005734719354, 176.4099615424925,
        144.77577266423688, 137.83729814221638, 127.8393513471905, 108.43487071765651, 101.7678237955058,
        83.74279738879831, 72.78289347220526, 68.41764138651662, 67.06907369418144, 58.882194188082764,
        53.56374590089697, 47.030838778081616, 46.27186100280124, 39.624951475667096, 39.06077387378444,
        37.023926363032, 35.037198385801965, 28.776841034879332, 22.140034661296983
    ])

    best_idx_est = np.argmin(rmse_est)
    best_idx_ref = np.argmin(rmse_ref)
    best_k_est = k[best_idx_est]
    best_k_ref = k[best_idx_ref]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_ylim(9,17)
    axes[0].plot(k, rmse_est, marker='o', color='tab:green')
    axes[0].plot(k, rmse_ref, marker='o', color='tab:orange')

    axes[0].vlines(
        best_k_est, min(rmse_est),max(rmse_est),
        color='darkgreen', linestyles='dashed',
        label=f'Matrix X from Est. Data, optimal k={best_k_est}'
    )
    axes[0].vlines(
        best_k_ref, min(rmse_est), max(rmse_est),
        color='darkorange', linestyles='dashed',
        label=f'Matrix X from Ref. Data, optimal k={best_k_ref}'
    )

    axes[0].set_xlabel("Rank k")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].set_title("MLR denoising (RMSE)")

    axes[1].plot(k, singular_values_est, marker='o', label='Matrix X from Est. Data')
    axes[1].plot(k, singular_values_ref, marker='o', label='Matrix X from Ref. Data')
    axes[1].set_ylabel("Singular value")
    axes[1].set_title("Singular values of Matrix X")
    axes[1].legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig("graphs/mlr_denoising_k=100.png")


if __name__ == '__main__':
    #plot_knn_k()
    plot_rf_k()
    #plot_mlr_k()