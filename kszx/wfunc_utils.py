"""
The ``kszx.window`` module contains functions for computing the window function.

This source file only contains a few functions right now. In the future I'll expand it
to include a "high-tech" computation of the window function.
"""

import numpy as np

from . import core

from .Box import Box


def compute_wapprox(box, fourier_space_footprints, rmax=0.03):
    """A crude approximation to the P(k) window function which neglects k-dependence and mixing.

    Given N "footprint" maps F_{ij}(x), this function computes an N-by-N matrix W_{ij} which
    gives the window function for a cross power spectrum on footprints i,j. (Note that since
    we're negelcting k-dependence and mixing, the window function doesn't have indices which
    correspond to k-bins.)
    
    Function arguments:

       - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
         See :class:`~kszx.Box` for more info.

       - ``fourier_space_footprints`` (array or list of arrays): single or multiple Fourier-space maps.

           - Case 1: If ``fourier_space_footprints`` is an array, then it represents a single Fourier-space map.
             (The array shape should be given by ``box.fourier_space_shape`` and the dtype should be ``complex``.)
    
           - Case 2: If ``fourier_space_footprints`` is a list of arrays, then it represents multiple
             Fourier-space maps. (Each map in the list should have shape ``box.fourier_space_shape``
             and dtype ``complex``.)
       
       - ``rmax`` (float): if correlation between any two footprints is < rmax, then an exception
         will be thrown. To disable this check, set rmax=0.

    The return value is either a scalar in case 1 (single Fourier-space map), or an array of shape
    ``(nmaps, nmaps)`` in case 2 (multiple maps).

    Sometimes a footprint is defined by a random catalog. Here is a reminder of how to make a
    Fourier-space map from a random catalog::

      box = ... # instance of class kszx.Box
      cosmo = ...  # instance of class kszx.Cosmology
      randcat = ...   # instance of class kszx.Catalog
      weights = ...   # 1-d array of length randcat.size
    
      xyz = randcat.get_xyz(cosmo, zcol_name='z')
      footprint = kszx.grid_points(box, xyz, rweights, kernel='cubic', fft=True, compensate=True)
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
