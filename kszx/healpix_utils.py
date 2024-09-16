import healpy
import warnings
import numpy as np

from . import utils
from . import io_utils

from .Catalog import Catalog


def read_map(filename):
    print(f'Reading {filename}')
    
    # Suppress superfluous warnings from healpy.fitsfunc.read_map():
    #   .../healpy/fitsfunc.py:391: UserWarning: NSIDE = 4
    #   .../healpy/fitsfunc.py:400: UserWarning: ORDERING = RING in fits file
    #   .../healpy/fitsfunc.py:428: UserWarning: INDXSCHM = IMPLICIT

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = healpy.read_map(filename, dtype=None, nest=False)
        
    # Convert from big-endian float (numpy dtype '>f4') to native float
    return np.asarray(m, float)


def write_map(filename, m):
    print(f'Writing {filename}')
    io_utils.mkdir_containing(filename)
        
    # Supply (dtype=m.dtype, overwrite=True).
    healpy.write_map(filename, m, dtype=m.dtype, overwrite=True)


def map_from_catalog(nside, gcat, weights=None, rcat=None, rweights=None, normalized=True):
    """Returns healpix map (float array with length 12*nside^2).

    The (rcat, rweights) arguments represent "randoms" to be subtracted.
    NOTE: the rweights are renormalized so that sum(rweights) = -sum(weights)!

    If normalized=True, then the output map includes a factor 1 / (pixel area).
    This normalization best represents a sum of delta functions f(x) = sum_j w_j delta^2(th_j).
    """
    
    assert isinstance(gcat, Catalog)
    assert isinstance(rcat, Catalog) or (rcat is None)

    if (rcat is None) and (rweights is not None):
        raise RuntimeError("kszx.healpix_utils.map_from_catalog(): 'rcat' arg is None, but 'rweights' arg is not None")

    weights = utils.asarray(weights, 'kszx.healpix_utils.map_from_catalog', 'weights', dtype=float, allow_none=True)
    rweights = utils.asarray(rweights, 'kszx.healpix_utils.map_from_catalog', 'rweights', dtype=float, allow_none=True)

    assert (weights is None) or (weights.ndim == 0) or (weights.shape == (gcat.size,))
    assert (rweights is None) or (rweights.ndim == 0) or (rweights.shape == (rcat.size,))
    
    npix = healpy.nside2npix(nside)
    ret = np.zeros(npix)

    ipix = healpy.ang2pix(nside, gcat.ra_deg, gcat.dec_deg, lonlat=True)
    wsum = utils.scattered_add(ret, ipix, weights)

    if rcat is not None:
        ipix = healpy.ang2pix(nside, rcat.ra_deg, rcat.dec_deg, lonlat=True)
        utils.scattered_add(ret, ipix, rweights, normalize_sum = -wsum)

    if normalized:
        ret /= healpy.nside2pixarea(nside)
    
    return ret


####################################################################################################


def read_alm(filename):
    print(f'Reading {filename}')

    alm, mmax = healpy.read_alm(filename, return_mmax=True)
    expected_nelts = ((mmax+1) * (mmax+2)) // 2

    if alm.shape != (expected_nelts,):
        raise RuntimeError(f'{filename}: expected lmax==mmax')

    return alm


def lmax_of_alm(alm):
    assert alm.ndim == 1
    
    s = alm.size
    lmax = int((2*s)**0.5) - 1
    
    if (s != ((lmax+1)*(lmax+2))//2) or (lmax < 0):
        raise RuntimeError(f'{alm.size=} is not of the form (lmax+1)(lmax+2)/2')
    
    return lmax


def degrade_alm(alm, dst_lmax):
    """I'm surprised this function isn't in healpy or pixell!"""
    
    src_lmax = lmax_of_alm(alm)
    assert 0 <= dst_lmax <= src_lmax

    ret = np.zeros(((dst_lmax+1)*(dst_lmax+2)) // 2, dtype=complex)
    d = 0
    s = 0
    
    for m in range(dst_lmax+1):
        dn = dst_lmax - m + 1
        sn = src_lmax - m + 1
        ret[d:(d+dn)] = alm[s:(s+dn)]
        d += dn
        s += sn

    return ret


def plot_alm(alm, nside=128, fwhm_deg=None, lmax=None):
    src_lmax = lmax_of_alm(alm)
    
    if (lmax is not None) and (lmax < src_lmax):
        alm = degrade_alm(alm, lmax)
        src_lmax = lmax

    if fwhm_deg is not None:
        assert fwhm_deg >= 0.0
        b = (fwhm_deg * np.pi/180.)**2 / (16. * np.log(2.))
        bl = np.exp(-b * np.arange(src_lmax+1)**2)
        alm = healpy.almxfl(alm, bl)

    m = healpy.alm2map(alm, nside)
    healpy.mollview(m)
