import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_actual: np.array, y_predicted: np.array) -> float:
    """
    Calculate RMSE
    :param y_actual:
    :param y_predicted:
    :return:
    """
    return np.sqrt(mean_squared_error(y_actual, y_predicted))