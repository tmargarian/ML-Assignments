# Keras uses the Sequential model for linear stacking of layers.
# That is, creating a neural network is as easy as (later)
# defining the layers!
from tensorflow.keras.models import Sequential
# Dropout 
from tensorflow.keras.layers import Dropout
# Everything we've talked about in class so far is referred to in
# Keras as a "dense" connection between layers, where every input
# unit connects to a unit in the next layer
# We will go over specific activation functions throughout the class.
from tensorflow.keras.layers import Dense
# SGD is the learning algorithm we will use
from tensorflow.keras.optimizers import SGD

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np



def build_one_output_model():
    model = Sequential()

    ### YOUR CODE HERE ###
    # Add a input hidden layer with appropriate input dimension
    # 1+ lines

    model.add(Dense(2, input_dim=2, activation = 'relu'))
   
    # Add a final output layer with 1 unit 
    # 1 line
   
    model.add(Dense(1, input_dim=2, activation = 'sigmoid'))

    ######################

    sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)  #Stochastic gradient descent
    model.compile(loss="binary_crossentropy", optimizer=sgd)
    return model


def build_classification_model():
    model = Sequential()

    ### YOUR CODE HERE ###
    # First add a fully-connected (Dense) hidden layer with appropriate input dimension
  
    model.add(Dense(10, input_dim=2, activation = 'relu'))
    
    # Now our second hidden layer 

    model.add(Dense(5, input_dim=10, activation = 'relu'))


    # Finally, add a readout layer
   
    model.add(Dense(2, input_dim=5, activation = 'softmax'))
    

    ######################

    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=["accuracy"])
    return model


def build_final_model():
    model = Sequential()
    ### YOUR CODE HERE ###
    
    model.add(Dense(70, input_dim=50, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(30, input_dim=70, activation = 'relu'))
    model.add(Dense(2, input_dim=30, activation = 'softmax'))

    ######################
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy

    return model

def logistic_regression_model():
    
    lin_regr = LogisticRegression()
    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = np.linspace(0.1, 1, 10)  

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)
    
    # Create grid search using 5-fold cross validation
    model = GridSearchCV(lin_regr, hyperparameters, scoring='f1', cv = 3, verbose = 2)
    return model


def random_forest_model():
    
    rf = RandomForestClassifier(random_state=26)
    
    param_grid = {
    'max_depth': [int(x) for x in range(20,40,5)], 
    'max_features': [int(x) for x in range(20,50,5)], 
    'min_samples_leaf': [int(x) for x in range(1,7,2)],
    'min_samples_split': [int(x) for x in range(2,8,2)],
    'n_estimators': [10, 30, 70]
    }
    model = GridSearchCV(estimator = rf, scoring='f1', param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    return model