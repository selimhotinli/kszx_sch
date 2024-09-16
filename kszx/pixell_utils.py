import numpy as np
import pixell.enmap
import pixell.enplot
import pixell.curvedsky

from . import utils
from . import io_utils

from .Catalog import Catalog


def read_map(filename):
    print(f'Reading {filename}')
    assert filename.endswith('.fits')
    return pixell.enmap.read_map(filename)


def write_map(filename, m):
    # I learned the hard way that you need this assert! If pixell.enmap.write_map() is
    # called with a numpy array (instead of a pixell ndmap), it will write a "time-bomb"
    # WCS which causes some pixell functions to fail later, with cryptic error messages.
    assert isinstance(m, pixell.enmap.ndmap)
    
    print(f'Writing {filename}')
    assert filename.endswith('.fits')
    io_utils.mkdir_containing(filename)
    pixell.enmap.write_map(filename, m)


def plot_map(m, downgrade, nolabels=True, filename=None, **kwds):
    """Thin wrapper around pixell.enplot(), just adding a few tweaks:

       - Make downgrade argument mandatory (usually needed in practice to avoid
         runaway heavyweight behavior). Suggest downgrade=40 for ACT.

       - Make nolabels=True the default (haven't figured out how to make labels
         look reasonable).

       - Add 'filename' argument to select between show/write.

    For more kwds, see:
      https://pixell.readthedocs.io/en/latest/reference.html#pixell.enplot.plot"""
    
    assert isinstance(m, pixell.enmap.ndmap)

    bunch = pixell.enplot.plot(m, downgrade=downgrade, nolabels=nolabels, **kwds)

    if filename is None:
        pixell.enplot.show(bunch)
    else:
        print(f'Writing {filename}')
        assert filename.endswith('.png')
        io_utils.mkdir_containing(filename)
        pixell.enplot.write(filename[:-4], bunch)


####################################################################################################


def _accum_catalog(ret, shape, wcs, catalog, weights, allow_outliers, normalize_sum=None):
    """Helper for map_from_catalog(). Returns sum of weights."""

    if not allow_outliers:
        # Note: if allow_outliers is False, then pixell_utils.ang2pix() returns (idec, ira).
        idec, ira = ang2pix(shape, wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=False)
    else:
        # Note: if allow_outliers is True, then pixell_utils.ang2pix() returns (idec, ira, mask).
        idec, ira, mask = ang2pix(shape, wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=True)
        idec, ira = idec[mask], ira[mask]
        wvec_flag = (weights is not None) and (weights.ndim > 0)
        weights = weights[mask] if wvec_flag else weights

    ret_1d = np.reshape(ret, (-1,))
    ix_1d = idec*shape[1] + ira

    # Return sum of weights, after applying outlier mask, but before applying 'normalize_sum'.
    return utils.scattered_add(ret_1d, ix_1d, weights, normalize_sum=normalize_sum)
    

def map_from_catalog(shape, wcs, gcat, weights=None, rcat=None, rweights=None, normalized=True, allow_outliers=True):
    """Returns a pixell.enmap.ndmap.
        
    The (rcat, rweights) arguments represent "randoms" to be subtracted.
    NOTE: the rweights are renormalized so that sum(rweights) = -sum(weights)!

    If 'allow_outliers' is True, then "out-of-bounds" galaxies (i.e. galaxies outside 
    the boundaries of the pixell map) will be silently discarded. If 'allow_outliers'
    is False, then out-of-bounds galaxies will raise an exception.

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

    ret = np.zeros(shape)
    wsum = _accum_catalog(ret, shape, wcs, gcat, weights, allow_outliers)

    if rcat is not None:
        _accum_catalog(ret, shape, wcs, rcat, rweights, allow_outliers, normalize_sum = -wsum)

    if normalized:
        ret /= pixell.enmap.pixsizemap(shape, wcs)
    
    return pixell.enmap.enmap(ret, wcs=wcs)


def eval_map_on_catalog(m, catalog, allow_outliers=True):
    """Returns 1-d array of map_vals if allow_outliers is False, or (map_vals, mask) if allow_outliers is True."""

    assert isinstance(m, pixell.enmap.ndmap)

    # pixell_utils.ang2pix() is defined later in this file.
    # It returns (idec,ira) if allow_outliers is False, or (idec,ira,mask) if allow_outliers is True.
    
    t = ang2pix(m.shape, m.wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=allow_outliers)
    idec, ira = t[0], t[1]
    map_vals = m[idec,ira]

    if allow_outliers:
        mask = t[2]
        map_vals = np.where(mask, map_vals, 0.)
        return map_vals, mask
    else:
        return map_vals
    

####################################################################################################


def alm2map(alm, shape, wcs):
    # FIXME map2alm() will blow up with a cryptic error message if it gets a bad WCS (e.g. zeroed).
    # Should find a general test for WCS badness, and add asserts throughout this file.
    
    alm = np.asarray(alm)
    assert alm.ndim == 1
    assert alm.dtype == complex

    m = pixell.enmap.enmap(np.zeros(shape), wcs=wcs)
    pixell.curvedsky.alm2map(alm, m)
    return m
    

def map2alm(m, lmax):
    assert isinstance(m, pixell.enmap.ndmap)
    assert lmax >= 0
    return pixell.curvedsky.map2alm(m, lmax=lmax)


####################################################################################################


def ang2pix(shape, wcs, ra_deg, dec_deg, allow_outliers=False):
    """Returns (idec,ira) if allow_outliers is False, or (idec,ira,mask) if allow_outliers is True.

    Roughly equivalent to healpy.ang2pix(..., latlon=True).
    Note that the argument ordering is (ra, dec), consistent with healpy but not pixell!

    Returns pair of integer-valued arrays (idec, ira), which contain indices
    with respect to the 'shape' tuple.

    The 'ra_deg' and 'dec_deg' array arguments should have the same shape, and
    the returned idec/ira arrays will also have this shape.

    Since pixell maps can be partial-sky, pixel indices can be out of bounds.
    In this case:
    
       - If allow_outliers=False, we throw an exception.

       - If allow_outliers=True, then we "clip" the returned idec/ira arrays
         so that they are in bounds, and return a boolean mask to indicate which
         values are valid.

    Note: if you are interested in this function in order to pixelize a Catalog,
    you may want to call Catalog.to_pixell() instead.
    """

    # pixell wants a single 'coords' array, with [dec,ra] converted to radians
    coords = np.stack(np.broadcast_arrays(dec_deg, ra_deg), axis=0)   # makes copy
    coords *= (np.pi/180.)

    # pixell.enmap.sky2pix() returns a single output array, where axis 0 is {idec,ira}.
    icoords = pixell.enmap.sky2pix(
        shape,
        wcs,
        coords,
        safe = True,      # pixell does not document this argument
        corner = False,   # indicates that integer coords correspond to pixel centers (not corners)
        bcheck = (not allow_outliers)   # pixell does not document this argument
    )

    icoords = np.asarray(np.round(icoords), dtype=int)

    icoords_clipped = np.clip(
        icoords,
        a_min = 0,
        a_max = np.reshape(shape, (2,)+(1,)*(icoords.ndim-1)) - 1   # note (-1) at end
    )

    mask = np.logical_and(
        icoords[0] == icoords_clipped[0], 
        icoords[1] == icoords_clipped[1]
    )

    idec, ira = icoords_clipped   # not icoords
    
    if allow_outliers:
        return idec, ira, mask
    elif not np.all(mask):
        raise RuntimeError('kszx.pixell_utils.ang2pix(): called with allow_outliers=False, and some points are out-of-bounds')
    else:
        return idec, ira
