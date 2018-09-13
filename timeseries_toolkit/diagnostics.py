import pandas as pd
import numpy as np
from scipy.signal import periodogram
from statsmodels.tsa.stattools import grangercausalitytests


def omega(series: pd.Series) -> float:
    """
    An estimator for the forecastability omaga(x_t) of a univariate time series x_t.

    Forecastability is defined as
    .. math::
        \Omega(x_t) = 1 - \frac{ - \int_{-\pi}^{\pi} f_x(\lambda)
            \log f_x(\lambda) d \lampbda }{\log 2 \pi} \in [0, 1]
    For white noise omega = 0; for a sum of sinusoids omega = 100.

    :param series: pandas.Series
    :return: float between 0 and 100. 0 means not forecastable (white noise);
    100 means perfectly forecastable (a sinusoid).
    """
    return (1 - spectral_entropy(series)) * 100


def spectral_entropy(series: pd.Series) -> float:
    """
    Calculate normalised spectral entropy of a time series
    :param series: pandas.Series
    :return: float between 0 and 100
    """
    _, psd = periodogram(series)
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log(psd_norm)) / np.log(len(psd_norm))