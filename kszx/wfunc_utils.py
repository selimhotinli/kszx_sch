"""
The ``kszx.wfunc_utils`` module contains functions for computing window functions.

This module is currently a placeholder -- it implements a crude approximation for the
window function, that can be used to roughly normalize $P(k)$. In the future I'll expand
it to include an accurate computation of the window function with k-bin mixing, spins, etc.
"""

import numpy as np

from . import core

from .Box import Box


def compute_wcrude(box, fourier_space_footprints, rmax=0.03):
    r"""A crude approximation to the $P(k)$ window function which neglects k-dependence and mixing.

    Given N "footprint" maps $F_{ij}(x)$, this function computes an N-by-N matrix $W_{ij}$ which
    gives the window function for a cross power spectrum on footprints i,j. (Note that since
    we're neglecting k-dependence and mixing, the window function doesn't have indices which
    correspond to k-bins.)

    For more details on what is computed, see the sphinx docs:
    
         https://kszx.readthedocs.io/en/latest/wfunc_utils.html#wcrude-details
    
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
         will be thrown. To disable this check, set ``rmax=0``.

    The return value is either a scalar in case 1 (single Fourier-space map), or an array of shape
    ``(nmaps, nmaps)`` in case 2 (multiple maps).

    Sometimes a footprint is defined by a random catalog. Here is a reminder of how to make a
    Fourier-space map from a random catalog::

      box = ...       # instance of class kszx.Box
      cosmo = ...     # instance of class kszx.Cosmology
      randcat = ...   # instance of class kszx.Catalog
      weights = ...   # 1-d array of length randcat.size
    
      xyz = randcat.get_xyz(cosmo, zcol_name='z')
      footprint = kszx.grid_points(box, xyz, rweights, kernel='cubic', fft=True, compensate=True)
    """

    map_list, multi_map_flag = core._parse_map_or_maps(box, fourier_space_footprints, 'kszx.window.wcrude')
    nfootprints = len(map_list)

    # FIXME revisit the issue of choosing K!
    K = 0.6 * box.knyq
    edges = np.array([ 0, K / 2**(1./box.ndim), K ])

    # use_dc=True is important here!!
    pk, counts = core.estimate_power_spectrum(box, map_list, edges, use_dc=True, return_counts=True)
    assert pk.shape == (nfootprints, nfootprints, 2)
    assert counts.shape == (2,)

    # Note factor (1/V_box), which normalizes sum_k -> int d^3k/(2pi)^3
    wcrude = (pk[:,:,0] - pk[:,:,1]) * counts[0] / box.box_volume
    
    for i in range(nfootprints):
        for j in range(i+1):
            r = wcrude[i,j] / np.sqrt(wcrude[i,i] * wcrude[j,j])
            if np.abs(r) < rmax:
                raise RuntimeError(f'kszx.window.wcrude(): correlation between footprints ({j,i}) is below threshold'
                                   + f' ({r=}, {rmax=}). This probably indicates an error somewhere. This check can be'
                                   + f' disabled entirely by passing rmax=0 to wcrude().')

    return wcrude if multi_map_flag else wcrude[0,0]


def compare_pk(pk1, pk2, noisy=True):
    """A utility function I wrote for testing: compares two P(k) arrays in a normalization-independent way.

    Function arguments:

      - ``pk1``, ``pk2``: arrays of either shape (nkbins,) or (nmaps,nmaps,nkbins).

    Returns a dimensionless number which is << 1 if the P(k) arrays are nearly equal.
    """
    
    pk1 = np.asarray(pk1)
    pk2 = np.asarray(pk2)

    if pk1.shape != pk2.shape:
        raise RuntimeError(f'Got {pk1.shape=} and {pk2.shape=}, expected equal shapes')

    if pk1.ndim == 1:
        pk1 = pk1.reshape((1,1,len(pk1)))
        pk2 = pk1.reshape((1,1,len(pk2)))
    
    elif (pk1.ndim != 3) or (pk1.shape[0] != pk1.shape[1]):
        raise RuntimeError(f'Got {pk1.shape=} and {pk2.shape=}, expected (nmaps,nmaps,nkbins)')

    nmaps, nkbins = pk1.shape[0], pk1.shape[2]
    w = np.ones((nmaps, nkbins))  # weights for power spectrum comparison

    for i in range(nmaps):
        if not np.all(pk1[i,i,:] > 0):
            raise RuntimeError(f'Not all auto power spectra pk1[{i},{i},:] were > 0')
        if not np.all(pk2[i,i,:] > 0):
            raise RuntimeError(f'Not all auto power spectra pk2[{i},{i},:] were > 0')
        w[i,:] = np.sqrt(pk1[i,i,:] + pk2[i,i,:])
    
    delta = np.abs(pk1-pk2) / (w[:,None,:] * w[None,:,:])

    if noisy and (nmaps > 1):
        print('compare_pk(): pairwise epsilon_max values')
        print(np.max(delta,axis=2))

    if noisy:
        print(f'compare_pk(): global epsilon_max value: {np.max(delta)}')
    
    return np.max(delta)
