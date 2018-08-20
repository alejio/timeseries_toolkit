from timeseries_toolkit import metrics
import numpy as np

def test_rmse():
    assert metrics.rmse(np.array([1,1]), np.array([0, 0])) == 1, \
        'RMSE should be one but got {}'.format(metrics.rmse(np.array([1,1]),
                                                            np.array([0, 0])))