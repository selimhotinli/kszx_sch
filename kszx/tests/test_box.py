from .. import Box
from . import helpers

import numpy as np

    
def test_k_component():
    print(f'test_k_component(): start')
    
    for iouter in range(100):
        box = helpers.random_box()
        axis = np.random.randint(0, box.ndim)
        zero_nyquist = (np.random.uniform() < 0.5)

        n = box.npix[axis]
        k = box.get_k_component(axis, zero_nyquist=zero_nyquist, one_dimensional=True)
        
        k2 = np.array([ i if (2*i <= n) else (i-n) for i in range(box.nk[axis]) ])
        k2 = k2 * (2*np.pi / box.boxsize[axis])
        
        if zero_nyquist and (n % 2 == 0):
            k2[n//2] = 0.

        eps = box.boxsize[axis] * np.max(np.abs(k-k2))
        assert eps < 1.0e-10
        
    print(f'test_k_component(): pass')


def test_r_component():
    print(f'test_r_component(): start')
    
    for iouter in range(100):
        box = helpers.random_box()
        axis = np.random.randint(0, box.ndim)
        r = box.get_r_component(axis, one_dimensional=True)
        r2 = np.linspace(box.lpos[axis], box.rpos[axis], box.npix[axis])
        eps = np.max(np.abs(r-r2))
        assert eps < 1.0e-10
        
    print(f'test_r_component(): pass')


def test_smallest_r():
    print(f'test_smallest_r(): start')

    for iouter in range(100):
        box = helpers.random_box()
        r = box.get_r()
        rmin = np.min(r)
        rmin2 = r[box._ix_smallest_r]
        eps = np.abs(rmin-rmin2)
        assert eps < 1.0e-10
        
    print(f'test_smallest_r(): pass')

