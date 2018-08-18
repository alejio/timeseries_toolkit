from timeseries_toolkit import transformations
import numpy as np

def test_rmse():
    assert transformations.rmse(np.array([1,1]), np.array([0, 0])) == 1, \
        'Failed '

# TODO: remaining tests