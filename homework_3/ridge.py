from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataset import load_features_and_labels
import numpy as np
import settings



def main():
    X, y = load_features_and_labels(settings.DatasetFile)
    model = make_pipeline(StandardScaler(), Ridge())
    gscv = GridSearchCV(
        model, param_grid={'ridge__alpha': np.logspace(-3, 3, 50)},
        scoring='neg_mean_squared_error', cv=5, return_train_score=True
    )
    gscv.fit(X, y)
    best_alpha = gscv.best_params_['ridge__alpha']
    alphas = gscv.cv_results_['param_ridge__alpha'].data
    mean_test = [abs(x) for x in gscv.cv_results_['mean_test_score']]
    print('alphas:', alphas)
    print("Mean test score:", mean_test)
    best_model = gscv.best_estimator_
    ridge = best_model.named_steps['ridge']
    print("Best alpha:", best_alpha)
    print(ridge.coef_)
    print(X.columns.to_numpy())
    feature_coefs = dict(zip(X.columns.to_numpy(), [round(abs(float(x)),3) for x in ridge.coef_]))
    sorted_feature_coefs = sorted(feature_coefs.items(), key=lambda x: x[-1],reverse=True)
    for k,v in sorted_feature_coefs:
        print(f'|{k}|{v}|')
    for i in range(0,12):
        print(f'{i+1}: {sum([x[-1] for x in sorted_feature_coefs[0:i+1]])}')
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    plt.plot(alphas, mean_test, marker='o')
    plt.vlines(best_alpha, min(mean_test),max(mean_test),
               color='black',linestyles='dashed',label=r'best $\lambda$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('mean squared error')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/feature_selection/ridge.png')





if __name__ == "__main__":
    main()