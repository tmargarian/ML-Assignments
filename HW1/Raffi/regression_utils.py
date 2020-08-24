import numpy as np
from sklearn.metrics import r2_score


def helloworld():
    print("Hello, world!")
    return


def simple_linear_regression_manually(x, y):
    """
    Implement simple linear regression below
    y: list of dependent variables
    x: list of independent variables
    return: b1- slope, b0- intercept
    """
    ### YOUR CODE HERE ###
    b1 = sum((y[i] - np.mean(y)) * (x[i] - np.mean(x)) for i in range(len(x))) / \
    sum((x[i] - np.mean(x)) ** 2 for i in range(len(x)))
    
    b0 = np.mean(y) - b1*np.mean(x)
    ######################
    return b1, b0


def simple_linear_regression_with_sklearn(x, y):
    """
    Implement simple linear regression below
    y: list of dependent variables
    x: list of independent variables
    return: b1- slope, b0- intercept
    """
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()  
    regressor.fit(x, y) #training the algorithm
    b0 = regressor.intercept_
    b1 = regressor.coef_
    return b1, b0


def multiple_regression(x, y):
    """
    x: np array of shape (n, p) where n is the number of samples
    and p is the number of features.
    y: np array of shape (n, ) where n is the number of samples
    return b: np array of shape (n, )
    """

    ### YOUR CODE HERE ###
    
    n = x.shape[0]
    x = np.hstack([np.ones([n,1]),x])
    b = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
    
    ######################
    return b


def predict_manually(x, b):
    ### YOUR CODE HERE ###
    
    n = x.shape[0]
    x = np.hstack([np.ones([n,1]),x])
    yhat = np.array(np.matmul(x,b))

    ######################
    return yhat


def predict_with_sklearn(x,y,x_test):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()  
    regressor.fit(x, y) #training the algorithm    
    y_pred = regressor.predict(x_test.reshape(-1,1))
    return y_pred

def calculate_r2(y, yhat):
    # y: np array of shape (n,) where n is the number of samples
    # yhat: np array of shape (n,) where n is the number of samples
    # yhat is calculated by predict()

    ### YOUR CODE HERE ###
    # calculate the residual sum of squares (rss) and total sum of squares (tss) 
    
    n = y.shape[0]
    rss = sum((y[i] - yhat[i]) ** 2 for i in range(n))
    tss = sum((y[i] - np.mean(y)) ** 2 for i in range(n))
    
    ######################

    r2 = 1.0 - rss/tss
    return r2

def calculate_adjusted_r2(y, yhat, k):
    n = y.shape[0]
    return 1 - (1 - calculate_r2(y, yhat) ** 2) * (n - 1)/(n - k - 1)


def check_r2(y, yhat):
    return np.allclose(calculate_r2(y, yhat), r2_score(y, yhat))
