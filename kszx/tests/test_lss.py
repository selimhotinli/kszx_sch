from .. import Box
from .. import lss
from .. import utils
from . import helpers

import itertools
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


def test_interpolation_gridding_consistency():
    print('test_interpolation_gridding_consistency(): start')
    
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
        
    print('test_interpolation_gridding_consistency(): pass')


####################################################################################################


# FIXME rename this
def test_simulate_gaussian():
    print('test_simulate_gaussian(): start')
    
    for _ in range(100):
        box = helpers.random_box()
        arr = lss.simulate_white_noise(box, fourier=True)
        arr2 = lss.fft_c2r(box, arr)
        arr2 = lss.fft_r2c(box, arr2)
        eps = np.max(np.abs(arr-arr2)) *  np.sqrt(box.pixel_volume)
        assert eps < 1.0e-10
        
    print('test_simulate_gaussian(): pass')


# FIXME turn into proper unit test
def monte_carlo_simulate_gaussian(npix, pixsize):
    box = Box(npix, pixsize)
    p_real = np.zeros(box.fourier_space_shape)
    p_fourier = np.zeros(box.fourier_space_shape)
        
    for nmc in itertools.count():
        x = lss.simulate_white_noise(box, fourier=False)
        x = lss.fft_r2c(box, x)
        p_real += np.abs(x)**2

        y = lss.simulate_white_noise(box, fourier=True)
        p_fourier += np.abs(y)**2

        if utils.is_perfect_square(nmc):
            r = p_fourier/p_real
            print(f'{nmc=} min_ratio={np.min(r)} max_ratio={np.max(r)}')
            
        
####################################################################################################


def _test_estimate_power_spectrum(box, kbin_delim, nmaps=None, use_dc=False):
    """Compares lss.estimate_power_spectrum() to a reference implementation.
    
    Note that nmaps=1 is slightly different from nmaps=None. 
    (The pk arrays have shape (1,1,nkbins) versus (nkbins,) in the two cases.)

    In principle, there is a small chance that this test can fail, if estimate_power_spectrum() 
    and the reference implementation put a k-mode in different bins, due to roundoff error. 
    """

    M = nmaps if (nmaps is not None) else 1
    maps = np.array([ lss.simulate_white_noise(box, fourier=True) for _ in range(M) ])
    
    map_arg = maps if (nmaps is not None) else maps[0]
    pk, bc = lss.estimate_power_spectrum(box, map_arg, kbin_delim, use_dc=use_dc, allow_empty_bins=True, return_counts=True)
    
    nbins = len(kbin_delim) - 1
    ref_pk = np.zeros((M,M,nbins))
    ref_bc = np.zeros(nbins, dtype=int)

    for ix in helpers.generate_indices(box.fourier_space_shape):
        i = np.array(ix, dtype=int)
            
        if (not use_dc) and np.all(i==0):
            continue

        k = np.array(i, dtype=float)
        k = np.minimum(k, box.npix - k)
        k *= (2*np.pi) / box.boxsize
        k = np.dot(k,k)**0.5
        b = np.searchsorted(kbin_delim, k, side='right')-1
        
        if (b < 0) or (b >= nbins):
            continue

        t = 1 if ((ix[-1]==0) or (2*ix[-1] == box.npix[-1])) else 2
        z = maps[(slice(None),) + ix]
        assert z.shape == (M,)
        
        ref_pk[:,:,b] += t * np.outer(z,z.conj()).real
        ref_bc[b] += t

    for b in range(nbins):
        if ref_bc[b] > 0:
            ref_pk[:,:,b] = (ref_pk[:,:,b] / ref_bc[b] / box.box_volume)

    if nmaps is None:
        ref_pk = ref_pk[0,0,:]
    
    tol = 1.0e-10 * (bc + ref_bc)**0.5

    assert np.all(bc == ref_bc)
    assert np.all(np.abs(pk - ref_pk) <= tol)


def test_estimate_power_spectrum():
    """Compares lss.estimate_power_spectrum() to a reference implementation.
    
    In principle, there is a small chance that this test can fail, if estimate_power_spectrum() 
    and the reference implementation put a k-mode in different bins, due to roundoff error. 
    """

    print('test_estimate_power_spectrum(): start')
    
    for iouter in range(100):
        box = helpers.random_box()
        use_dc = (np.random.uniform() < 0.5)

        # Note! Currently assuming nmaps=4 is max value supported by kernel.
        # If the kernel changes in the future, make sure to update the line below.
        nmaps = np.random.randint(1,5) if (np.random.uniform() < 0.95) else None
        # print(f'{nmaps=}')
        
        # Generate random kbin_delim
        nbins = np.random.randint(2, 11)
        kmax = 1.1 * np.sqrt(box.ndim) * box.knyq
        kbin_delim = np.random.uniform(0.0, kmax, nbins+1)
        kbin_delim = np.sort(kbin_delim)
        kbin_delim += np.linspace(0.0, 1.0e-10 * kmax, nbins+1)
        kbin_delim[0] = kbin_delim[0] if (np.random.uniform() < 0.5) else 0.0

        _test_estimate_power_spectrum(box, kbin_delim, nmaps=1, use_dc=use_dc)
        
    print('test_estimate_power_spectrum(): pass')
