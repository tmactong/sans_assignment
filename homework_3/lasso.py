from sklearn.linear_model import LassoLarsCV
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataset import load_features_and_labels
import settings
import time


def main():
    X, y = load_features_and_labels(settings.DatasetFile)
    start_time = time.time()
    model = make_pipeline(StandardScaler(), LassoLarsCV(cv=20)).fit(X, y)
    fit_time = time.time() - start_time
    lasso = model[-1]
    print(lasso.coef_)
    feature_names = X.columns.to_numpy()
    print(feature_names)
    print(f'lambda: {lasso.alpha_}')
    plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ":")
    plt.semilogx(
        lasso.cv_alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label=r"$\lambda$ CV")

    # plt.ylim(ymin, ymax)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Mean square error")
    plt.legend()
    # plt.title(f"Mean square error on each fold: Lars (train time: {fit_time:.2f}s)")
    # plt.show()
    plt.savefig("graphs/feature_selection/lasso.png")



if __name__ == "__main__":
    main()