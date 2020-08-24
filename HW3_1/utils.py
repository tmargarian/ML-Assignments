import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def visualize_2d_data(df, h=0.02):
    y = df["y"].values
    x = df.drop("y", axis=1).values
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    f = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.tight_layout()
    ax = f.get_axes()
    return f, ax[0]