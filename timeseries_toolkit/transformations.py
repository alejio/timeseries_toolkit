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

def target_transform(y: np.array, increment: float=0.01) -> np.array:
    """
    Transform non-negative array to R using np.log
    :param y:
    :return:
    """
    return np.log(y + increment)

def target_inverse_transform(y_trn: np.array, increment: float=0.01) -> np.array:
    """
    Inverse transform of array in R to non-negative
    :param y_trn:
    :param increment:
    :return:
    """
    return np.exp(y_trn) - increment