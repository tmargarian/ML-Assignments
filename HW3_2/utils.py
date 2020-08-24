import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


def get_colors(y):
    colors = ('red', 'blue', 'green', 'gray', 'cyan', 'black', 'purple')
    cmap = ListedColormap(colors[:len(set(y))])
    return cmap


def close_enough(x, y):
    return np.allclose(x, y)


def get_accuracy(model, x, y):
    pred = model.predict_classes(x)
    return accuracy_score(y, pred)


def lenses_dataset():
    n_samples=200
    noise=0.1
    seed = 23
    return make_moons(n_samples=n_samples, noise=noise,
                               random_state=seed)


def make_classification_dataset(n_samples, n_features,
                                n_classes=2, n_informative=None,
                                noise=0.01, seed=None):
    if n_informative is None:
        n_informative = n_features
    return make_classification(n_samples=n_samples, n_features=n_features,
                               n_redundant=0, n_repeated=0,
                               n_informative=n_informative, n_classes=n_classes,
                               flip_y=noise, shuffle=True, random_state=seed)


def visualize_2d_classification(model, x, y, h=0.05):
    # set up plotting grid
    xx1, xx2 = np.meshgrid(np.arange(x[:,0].min(), x[:,0].max(), h),
                           np.arange(x[:,1].min(), x[:,1].max(), h))
    grid = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = model.predict_classes(grid)
    Z = np.array(Z)
    Z = 1-Z.reshape(xx1.shape)  # for display purposes
    cmap = get_colors(y)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    f = plt.figure()
    # Decision boundary drawn on training set
    plt.contourf(xx1, xx2, Z, alpha=0.8)
    plt.scatter(x[:,0], x[:,1],
                c=y, cmap=cmap)

    plt.tight_layout()
    ax = f.get_axes()
    return f, ax[0]


def create_and_write_datasets(n_features, noise=0.01, seed=None):
    x, y = make_classification_dataset(
        200, n_features, n_classes=2, n_informative=None,
        noise=noise, seed=seed)

    df = pd.DataFrame(x, columns=["x{}".format(i+1) for i in range(n_features)])
    df["y"] = y
    df[:100].to_csv("train_{}features.csv".format(n_features), index=False)
    df[100:].to_csv("test_{}features.csv".format(n_features), index=False)
    return df


def create_datasets():
    # df = create_and_write_datasets(2, noise=0.4, seed=20)
    # df = create_and_write_datasets(50, noise=0.1, seed=40)
    # test_df = pd.read_csv("test_50features.csv")
    # test_df.drop("y", axis=1).to_csv("test_50features.csv", index=False)
    pass

def check_score():
    pass
