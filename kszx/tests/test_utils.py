"""This file is currently extremely incomplete! Some day I'll test more utility functions."""

import numpy as np

from .. import utils


def test_contract_axis():
    print('test_contract_axis(): start')
    
    for _ in range(100):
        ndim = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, size=(ndim,))
        axis = np.random.randint(0, ndim)
        arr = np.random.normal(size=shape)
        weights = np.random.normal(size=shape[axis])
        
        ret = utils.contract_axis(arr, weights, axis)

        # Alternate implementation of utils.contract_axis().
        ret2 = 0
        for i,w in enumerate(weights):
            ret2 = ret2 + w * np.take(arr, i, axis=axis)

        epsilon = np.max(np.abs(ret-ret2))
        assert epsilon < 1.0e-10

    print('test_contract_axis(): pass')
    
