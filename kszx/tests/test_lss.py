from .. import Box
from .. import lss
from . import helpers

import numpy as np


class RandomPoly:
    def __init__(self, degree, lpos, rpos):
        assert lpos.shape == rpos.shape
        assert lpos.ndim == 1
        assert degree >= 1

        self.ndim = len(lpos)
        self.degree = 1
        self.lpos = lpos
        self.rpos = rpos
        self.cpos = (rpos + lpos) / 2.
        self.coeffs = np.zeros((self.ndim, degree+1))

        for axis in range(self.ndim):
            dx = (rpos[axis] - lpos[axis]) / 2.
            for i in range(degree+1):
                self.coeffs[axis,i] = np.random.uniform(-1.0, 1.0) / dx**i


    def _eval_axis(self, x, axis):
        t = x - self.cpos[axis]
        tpow = np.ones_like(t)
        ret = np.zeros_like(t)

        for i in range(self.degree+1):
            ret += self.coeffs[axis,i] * tpow
            tpow *= t

        return ret
    
            
    def eval_points(self, points):
        assert points.ndim == 2
        assert points.shape[-1] == self.ndim
        
        npoints = points.shape[0]
        ret = np.ones(npoints)

        for axis in range(self.ndim):
            ret *= self._eval_axis(points[:,axis], axis)

        return ret
        
        
    def eval_grid(self, shape):
        assert len(shape) == self.ndim
        ret = np.ones(shape)

        for axis in range(self.ndim):
            x = np.linspace(self.lpos[axis], self.rpos[axis], shape[axis])
            y = self._eval_axis(x, axis)
            s = np.concatenate((np.ones(axis,dtype=int), (-1,), np.ones(self.ndim-axis-1,dtype=int)))
            ret *= np.reshape(y, s)

        return ret


####################################################################################################


def test_interpolation():
    print('test_interpolation(): start')
    
    for _ in range(100):
        # Currently, lss.interpolate_points() only supports CIC.
        kernel, degree = ('cic', 1)

        # Currently, lss.interpolate_point() only supports ndim=3.
        box = helpers.random_box(ndim=3, nmin=degree+1)
        poly = RandomPoly(degree=degree, lpos=box.lpos, rpos=box.rpos)
        ndim = box.ndim

        npoints = np.random.randint(100, 200)
        pad = (degree - 1 + 1.0e-7) * (box.pixsize/2.)
        points = np.random.uniform(box.lpos+pad, box.rpos-pad, size=(npoints,ndim))
        
        grid = poly.eval_grid(box.npix)
        exact_vals = poly.eval_points(points)
        interpolated_vals = lss.interpolate_points(box, grid, points, kernel)

        epsilon = helpers.compare_arrays(exact_vals, interpolated_vals)
        assert epsilon < 1.0e-12
        
    print('test_interpolation(): pass')


def test_gridding():
    print('test_gridding(): start')
    
    for _ in range(100):
        # Currently, lss.interpolate_points() only supports CIC.
        kernel, degree = ('cic', 1)

        # Currently, lss.interpolate_point() only supports ndim=3.
        box = helpers.random_box(ndim=3, nmin=degree+1)
        ndim = box.ndim

        npoints = np.random.randint(100, 200)
        pad = (degree - 1 + 1.0e-7) * (box.pixsize/2.)
        points = np.random.uniform(box.lpos+pad, box.rpos-pad, size=(npoints,ndim))

        g = np.random.normal(size=box.npix)  # random grid
        w = np.random.normal(size=npoints)   # random weights
        
        Ag = lss.interpolate_points(box, g, points, kernel)
        Aw = np.zeros(shape=box.npix)
        lss.grid_points(box, Aw, points, kernel, weights=w)

        dot1 = np.dot(w,Ag)
        dot2 = helpers.map_dot_product(box,Aw,g)
        
        den = np.dot(w,w) * np.dot(Ag,Ag)
        den += helpers.map_dot_product(box,g,g) * helpers.map_dot_product(box,Aw,Aw)

        epsilon = np.abs(dot1-dot2) / den**(0.5)
        assert epsilon < 1.0e-12
        
    print('test_gridding(): pass')
