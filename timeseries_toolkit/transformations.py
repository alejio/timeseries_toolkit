import numpy as np


def target_transform(y: np.array, increment: float=0.01) -> np.array:
    """
    Transform non-negative array to R using np.log
    :param y: np.array
    :param increment: float
    :return:
    """
    return np.log(y + increment)


def target_inverse_transform(y_trn: np.array, increment: float=0.01) -> np.array:
    """
    Inverse transform of array in R to non-negative
    :param y_trn: np.array
    :param increment: float
    :return:
    """
    return np.exp(y_trn) - increment