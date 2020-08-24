import numpy as np
from sklearn.metrics import r2_score


def helloworld():
    #print("Hello, world!")
    print("uncomment me")
    return


def simple_linear_regression(x, y):
    """
    Implement simple linear regression below
    y: list of dependent variables
    x: list of independent variables
    return: b1- slope, b0- intercept
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = sum( (x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(len(y)))
    b1 = numerator / denominator
    b0 = y_mean - b1 * x_mean

    ######################
    return b1, b0


def multiple_regression(x, y):
    """
    x: np array of shape (n, p) where n is the number of samples
    and p is the number of features.
    y: np array of shape (n, ) where n is the number of samples
    return b: np array of shape (n, )
    """


    ### YOUR CODE HERE ###

    X = x.values
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    ######################
    return b


def predict(x, b):
    ### YOUR CODE HERE ###



    ######################
    return yhat


def calculate_r2(y, yhat):
    # y: np array of shape (n,) where n is the number of samples
    # yhat: np array of shape (n,) where n is the number of samples
    # yhat is calculated by predict()

    ### YOUR CODE HERE ###
    # calculate the residual sum of squares (rss) and total sum of squares (tss)
    

    
    ######################

    r2 = 1.0 - rss/tss
    return r2


def check_r2(y, yhat):
    return np.allclose(calculate_r2(y, yhat), r2_score(y, yhat))
