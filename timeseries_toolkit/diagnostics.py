import pandas as pd
import numpy as np
from scipy.signal import periodogram

def omega(series: pd.Series) -> float:

    h_spectral = spectral_entropy(series)
    omega = (1 - h_spectral) * 100
    return omega

def spectral_entropy(series: pd.Series):

    _, psd = periodogram(series)
    psd_norm = psd/np.sum(psd)
    spectral_entropy = np.sum(psd_norm * np.log(psd_norm)) / \
                       np.log(len(psd_norm))
    return -1 * spectral_entropy