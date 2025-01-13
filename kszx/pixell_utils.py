import numpy as np
import pixell.enmap
import pixell.enplot
import pixell.curvedsky

from . import utils
from . import io_utils

from .Catalog import Catalog


def read_map(filename):
    print(f'Reading {filename}\n', end='')
    assert filename.endswith('.fits')
    return pixell.enmap.read_map(filename)


def write_map(filename, m):
    # I learned the hard way that you need this assert! If pixell.enmap.write_map() is
    # called with a numpy array (instead of a pixell ndmap), it will write a "time-bomb"
    # WCS which causes some pixell functions to fail later, with cryptic error messages.
    assert isinstance(m, pixell.enmap.ndmap)
    
    print(f'Writing {filename}\n', end='')
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
        print(f'Writing {filename}\n', end='')
        assert filename.endswith('.png')
        io_utils.mkdir_containing(filename)
        pixell.enplot.write(filename[:-4], bunch)


####################################################################################################


def _accum_catalog(ret, shape, wcs, catalog, weights, allow_outliers, normalize_sum=None):
    """Helper for map_from_catalog(). Returns sum of weights."""

    if not allow_outliers:
        # Note: if allow_outliers is False, then pixell_utils.ang2pix() returns (idec, ira).
        msg = "kszx.pixell_utils.map_from_catalog(): some points were out-of-bounds, you may want allow_outliers=True"
        idec, ira = ang2pix(shape, wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=False, outlier_errmsg=msg)
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


def eval_map_on_catalog(m, catalog, pad=None, return_mask=False):
    """Evaluates pixell map 'm' at (ra,dec) values from catalog, and returns a 1-d array.

    Since pixell maps can be partial-sky, some of the (ra,dec) pairs may be "outliers",
    i.e. outside the map footprint. If pad=None (the default), this raises an exception.
    If 'pad' is specified (e.g. pad=0.0) then outlier values are replaced by 'pad'.

    If return_mask=False (the default), returns a 1-d array 'map_vals'.

    If return_mask=True, returns (map_vals, mask), where 'mask' is a 1-d boolean array which
    is True for non-outliers, False for outliers.
    """

    assert isinstance(m, pixell.enmap.ndmap)
    assert isinstance(catalog, Catalog)

    if return_mask and (pad is None):
        raise RuntimeError("kszx.pixell_utils.eval_map_on_catalog(): if return_mask=True, then 'pad' must be specified")
    if pad is not None:
        pad = float(pad)
    
    # pixell_utils.ang2pix() is defined later in this file.
    # It returns (idec,ira) if allow_outliers is False, or (idec,ira,mask) if allow_outliers is True.

    msg = "kszx.pixell_utils.eval_map_on_catalog(): some points are out-of-bounds, you may want to specify the 'pad' arugment"
    t = ang2pix(m.shape, m.wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=(pad is not None), outlier_errmsg=msg)
    
    idec, ira = t[0], t[1]
    mask = t[2] if (len(t) > 2) else None
    map_vals = m[idec,ira]

    if pad is not None:
        map_vals = np.where(mask, map_vals, pad)

    return (map_vals, mask) if return_mask else map_vals
    

####################################################################################################


def alm2map(alm, shape, wcs):
    # FIXME map2alm() will blow up with a cryptic error message if it gets a bad WCS (e.g. zeroed).
    # Should find a general test for WCS badness, and add asserts throughout this file.
    
    alm = np.asarray(alm)
    assert alm.ndim == 1
    assert (alm.dtype == complex) or (alm.dtype == np.complex64)

    m = pixell.enmap.enmap(np.zeros(shape), wcs=wcs)
    pixell.curvedsky.alm2map(alm, m)
    return m
    

def map2alm(m, lmax):
    assert isinstance(m, pixell.enmap.ndmap)
    assert lmax >= 0
    return pixell.curvedsky.map2alm(m, lmax=lmax)


####################################################################################################


def ang2pix(shape, wcs, ra_deg, dec_deg, allow_outliers=False, outlier_errmsg=None):
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
    elif np.all(mask):
        return idec, ira

    if outlier_errmsg is None:
        outlier_errmsg = 'kszx.pixell_utils.ang2pix(): called with allow_outliers=False, and some points are out-of-bounds'
    raise RuntimeError(outlier_errmsg)


####################################################################################################


def uK_arcmin_from_ivar(ivar, max_uK_arcmin=1.0e6):
    r"""Given an inverse variance map (ivar), returns associated RMS map (in uK-arcmin).

    Based on Mat's orphics library:
       https://github.com/msyriac/orphics/blob/master/orphics/maps.py

    **Note:** currently uses pixell maps, but could easily be modified to allow healpix
    maps -- let me know if this would be useful.

    Function args:
    
      - ``ivar`` (pixell map): inverse variance map, units (uK)^{-2}.
        (For example, the return value from :func:`~kszx.act.read_ivar()`.)

      - ``max_uK_arcmin`` (float): max allowed value in the return map (prevents
        dividing by zero).

    Return value is a pixell map with units uK-arcmin.

    Equivalent to::
    
       ps = ivar.pixsizemap()
       uK_arcmin = sqrt(ps/ivar) * (60*180/np.pi)
       return np.minimum(uK_arcmin, max_uK_arcmin)

    but regulated to avoid dividing by zero.
    """

    assert isinstance(ivar, pixell.enmap.ndmap)
    assert max_uK_arcmin > 0

    a = 60*180/np.pi
    ps = ivar.pixsizemap()
    epsilon = 0.9 * np.min(ps) * (a/max_uK_arcmin)**2
    
    uK_arcmin = a * np.sqrt(ps / np.maximum(ivar,epsilon))
    return np.minimum(uK_arcmin, max_uK_arcmin)


def fkp_from_ivar(ivar, cl0, normalize=True, return_wvar=False):
    r"""Given an inverse variance map (ivar), returns associated FKP weighting.

    $$\begin{align}
    W(\theta) &= \frac{1}{C_l^{(0)} + N(\theta)} \\
    N(\theta) &\equiv \frac{\mbox{Pixel area}}{\mbox{ivar}(\theta)} \hspace{1cm} \mbox{``Local'' noise power spectrum}
    \end{align}$$

    **Note:** currently uses pixell maps, but could easily be modified to allow healpix
    maps -- let me know if this would be useful.

    Function args:

      - ``ivar`` (pixell map): inverse variance map, units (uK)^{-2}.
        (For example, the return value from :func:`~kszx.act.read_ivar()`.)

      - ``cl0`` (float): FKP weight function parameter $C_l^{(0)}$.

          - Intuitively, ``cl0`` = "fiducial signal $C_l$ at wavenumber $l$ of interest".
          - ``cl0=0`` corresponds to inverse noise weighting.
          - Large ``cl0`` corresponds to uniform weighing (but ``cl0=np.inf`` won't work).
          - I usually use ``cl0 = 0.01`` for plotting all-sky CMB temperature maps.
          - For kSZ filtering, ``cl0=3e-5`` is a reasonable choice.

      - ``normalize`` (boolean): if True, then we normalize the weight function 
        so that $\max(W(\theta))=1$.

      - ``return_wvar`` (boolean): if True, then we return $W(\theta) / ivar(\theta)$,
        instead of returning $W(\theta)$.
    
    Returns a pixell map.

    In implementation, in order to avoid divide-by-zero for ivar=0, we compute
    $W(\theta)$ equivalently as:
    
    $$W(\theta) = \frac{\mbox{ivar}(\theta)}{(\mbox{pixel area}) + C_l^{(0)} \mbox{ivar}(\theta)}$$
    """
    
    assert isinstance(ivar, pixell.enmap.ndmap)
    assert ivar.ndim == 2
    assert np.all(ivar >= 0.0)
    assert cl0 >= 0.0

    wvar = 1.0 / (ivar.pixsizemap() + cl0 * ivar)
    w = wvar * ivar
    
    wmax = np.max(w)
    assert wmax > 0.0

    ret = wvar if return_wvar else w
    
    if normalize:
        ret /= wmax

    return ret
