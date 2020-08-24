import numpy as np

def accuracy(truth_values, predicted_values):
    """
    Computes accuracy
    :param truth_values: true values
    :param predicted_values: predictions

    :return: accuracy
    """
    ### YOUR CODE HERE ###
    
    correct_predictions = truth_values[truth_values == predicted_values].shape[0]
    total_predictions = truth_values.shape[0]
    accuracy = correct_predictions / total_predictions

    ######################

    return accuracy

def precision(truth_values, predicted_values):
    """
    Computes precision
    :param truth_values: true values
    :param predicted_values: predictions

    :return: precision
    """
    ### YOUR CODE HERE ###
    
    n = truth_values.shape[0]
    
    prediction_matrix = np.vstack((truth_values,predicted_values)).T 
    
    FP_con1 = prediction_matrix[:,0] == np.zeros(n)
    FP_con2 = prediction_matrix[:,1] == np.ones(n)
    FP = prediction_matrix[FP_con1 & FP_con2].shape[0]
    
    TP_con1 = prediction_matrix[:,0] == np.ones(n)
    TP_con2 = prediction_matrix[:,1] == np.ones(n)
    TP = prediction_matrix[TP_con1 & TP_con2].shape[0]
    
    precision = TP / (TP + FP)
    
    ######################

    return precision


def recall(truth_values, predicted_values):
    """
    Computes recall
    :param truth_values: true values
    :param predicted_values: predictions

    :return: recall
    """
    ### YOUR CODE HERE ###
    
    n = truth_values.shape[0]
    
    prediction_matrix = np.vstack((truth_values,predicted_values)).T 

    FN_con1 = prediction_matrix[:,1] == np.zeros(n)
    FN_con2 = prediction_matrix[:,0] == np.ones(n)
    FN = prediction_matrix[FN_con1 & FN_con2].shape[0]
    
    TP_con1 = prediction_matrix[:,0] == np.ones(n)
    TP_con2 = prediction_matrix[:,1] == np.ones(n)
    TP = prediction_matrix[TP_con1 & TP_con2].shape[0]
 
    recall = TP / (TP + FN)
    
    ######################

    return recall

