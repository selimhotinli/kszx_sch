import numpy as np
import pixell.enmap
import pixell.enplot
import pixell.curvedsky

from . import io_utils


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
    io_utils.mkdir_contaning(filename)
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


def alm2map(alm, shape, wcs):
    # FIXME map2alm() will blow up with a cryptic error message if it gets a bad WCS (e.g. zeroed).
    # Should find a general test for WCS badness, and add asserts throughout this file.
    
    alm = np.asarray(alm)
    assert alm.ndim == 1
    assert alm.dtype == complex

    m = pixell.enmap.enmap(np.zeros(shape), wcs=wcs)
    pixell.curvedsky.alm2map(alm, m)
    return m
    

def map2alm(m, lmax, *, weight):
    """The 'weight' argument can be either 'ring', 'area', or 'unit'."""
    
    assert isinstance(m, pixell.enmap.ndmap)
    assert lmax >= 0

    nalm = ((lmax+1)*(lmax+2)) // 2
    alm = np.zeros(nalm, dtype=complex)

    if weight == 'ring':
        pixell.curvedsky.map2alm(m, alm=alm)
    elif weight == 'area':
        m = m * pixell.enmap.pixsizemap(m.shape, m.wcs)
        pixell.curvedsky.alm2map_adjoint(m, alm=alm)
    elif weight == 'unit':
        pixell.curvedsky.alm2map_adjoint(m, alm=alm)
    else:
        raise RuntimeError(f"kszpipe.pixell_utils.map2alm: {weight=} invalid (expected 'ring', 'area', or 'unit').")
        
    return alm


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
