from timeseries_toolkit import transformations
import numpy as np


def test_target_transform():
    assert transformations.target_transform(0, np.exp(1)) == 1, \
        "Expected is 1 but got {}".format(transformations.target_transform(0, np.exp(1)))

def test_inverse_target_transform():
    assert transformations.target_inverse_transform(1, np.exp(1)) == 0, \
        "Expected is 0 but got {}".format(transformations.target_inverse_transform(1, np.exp(1)))