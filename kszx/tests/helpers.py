from .. import Box

import numpy as np

    
def compare_arrays(arr1, arr2):
    assert arr1.shape == arr2.shape
    assert arr1.dtype == arr2.dtype

    t = arr1-arr2
    num = np.vdot(t,t)
    den = np.vdot(arr1,arr1) + np.vdot(arr2,arr2)
    
    return np.sqrt(num/den) if (den > 0.0) else 0.0


def random_shape(ndim=None, nmin=1):
    if ndim is None:
        ndim = np.random.randint(1,4)
    
    ret = np.zeros(ndim, dtype=int)
    nmax = int(10000 ** (1./ndim))
    assert nmax >= (nmin+1)
    
    for d in range(ndim):
        if np.random.uniform() < 0.1:
            ret[d] = nmin
        elif np.random.uniform() < 0.1:
            ret[d] = nmin+1
        else:
            ret[d] = np.random.randint(nmin, nmax+1)

    return ret


def random_box(ndim=None, nmin=2):
    npix = random_shape(ndim, nmin)
    pixsize = np.random.uniform(1.0, 10.0)
    cpos = np.random.uniform(-1000, 1000, size=len(npix))
    return Box(npix, pixsize, cpos)
