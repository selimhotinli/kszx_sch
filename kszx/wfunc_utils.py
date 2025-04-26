"""
The ``kszx.window`` module contains functions for computing the window function.

This source file only contains a few functions right now. In the future I'll expand it
to include a "high-tech" computation of the window function.
"""

import numpy as np

from . import core

from .Box import Box


def compute_wapprox(box, fourier_space_footprints, rmax=0.03):
    """
    Approximation to true window function which neglects k-dependence, spins!

    Case 1: 'fourier_space_footprints' is a single Fourier-space map.
      Returns a scalar.

    Case 2: 'fourier_space_footprints' is a length-nfootprints iterable returning
      Fourier-space maps. Returns an array of shape (nfootprints, nfootprints).
    
    Reminder::

      box = ... # instance of class kszx.Box
      cosmo = ...  # instance of class kszx.Cosmology
      randcat = ...   # instance of class kszx.Catalog
      weights = ...   # 1-d array of length randcat.size
    
      xyz = randcat.get_xyz(cosmo, zcol_name='z')
      footprint = kszx.grid_points(box, xyz, rweights, kernel='cubic', fft=True)
      kszx.apply_kernel_compensation(box, footprint, kernel='cubic')
    """

    map_list, multi_map_flag = core._parse_map_or_maps(box, fourier_space_footprints, 'kszx.window.wapprox')
    nfootprints = len(map_list)

    # FIXME revisit the issue of choosing K!
    K = 0.6 * box.knyq
    edges = np.array([ 0, K / 2**(1./box.ndim), K ])

    # use_dc=True is important here!!
    pk, counts = core.estimate_power_spectrum(box, map_list, edges, use_dc=True, return_counts=True)
    assert pk.shape == (nfootprints, nfootprints, 2)
    assert counts.shape == (2,)

    # Note factor (1/V_box), which normalizes sum_k -> int d^3k/(2pi)^3
    wapprox = (pk[:,:,0] - pk[:,:,1]) * counts[0] / box.box_volume
    
    for i in range(nfootprints):
        for j in range(i+1):
            r = wapprox[i,j] / np.sqrt(wapprox[i,i] * wapprox[j,j])
            if np.abs(r) < rmax:
                raise RuntimeError(f'kszx.window.wapprox(): correlation between footprints ({j,i}) is below threshold'
                                   + f' ({r=}, {rmax=}). This probably indicates an error somewhere. This check can be'
                                   + f' disabled entirely by passing rmax=0 to wapprox().')

    return wapprox if multi_map_flag else wapprox[0,0]


def scale_wapprox(wapprox, weights, index_map=None):
    # Argument checking starts here.
    wapprox = np.asarray(wapprox)
    weights = np.asarray(weights)
    index_map = np.asarray(index_map) if (index_map is not None) else None

    if wapprox.ndim == 0:
        wapprox = np.reshape(wapprox, (1,1))
    elif (wapprox.ndim != 2) or (wapprox.shape[0] != wapprox.shape[1]):
        raise RuntimeError(f'Got {wapprox.shape=}, expected (N,N) or scalar')

    if index_map is None:
        index_map = np.arange(wapprox.shape[0])    
    if (index_map is not None) and (index_map.ndim != 1):
        raise RuntimeError(f'Got {index_map.shape=}, expected 1-d array or None')

    nin = wapprox.shape[0]
    nout = index_map.shape[0]

    if weights.shape != (nout,):
        raise RuntimeError(f'Got {weights.shape=}, expected 1-d array of length {nout}')

    # Argument checking finished -- now for the one-line function body :)
    return wapprox[index_map,:][:,index_map] * np.outer(weights,weights)
