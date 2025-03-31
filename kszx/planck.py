"""Currently, the only Planck data product that I'm using is the galmask, so there's not much here."""

import os
import fitsio
import healpy
import numpy as np

from . import io_utils

    
def read_hfi_galmask(sky_percentage, apodization=0, dtype=None, download=False):
    """Returns an nside=2048 healpix map in RING ordering and Galactic coordinates.

    Default dtype is either uint8 or float32 (depending on whether apodization > 0), 
    but this can be changed with the ``dtype`` argument.

    Allowed ``sky_percentage`` values: 20, 40, 60, 70, 80, 90, 97, 99
    Allowed ``apodization`` values: 0, 2, 5 (degrees)
    If ``download`` is True, then data files will be auto-downloaded.

    Note that we use Planck release 2, since release 3 doesn't seem to have HFI
    foreground masks (it does have other masks).
    
    To apply a Planck mask to ACT data, you'll need to rotate/pixellize the mask::

       pixell_mask = pixell.reproject.healpix2map(
          healpix_mask, shape, wcs, 
          rot='gal,equ',                # NOTE coordinate rotation!!
          method='spline', order=0)     # NOTE method='spline', not method='harm'!
    """

    assert sky_percentage in [ 20, 40, 60, 70, 80, 90, 97, 99 ]
    
    abspath = _hfi_galmask_filename(apodization, download, dlfunc='kszx.planck.read_hfi_galmask')
    col_name = f'GAL0{sky_percentage}'
    print(f'Reading {abspath} ({col_name=})\n', end='')
    
    with fitsio.FITS(abspath) as f:
        if col_name not in f[1].get_colnames():
            raise RuntimeError(f'{abspath}: column name {col_name} not found')
        mask = f[1][col_name].read()

    # Convert from Healpix NESTED (in fits file) to RING (assumed by pixell)
    mask = healpy.pixelfunc.reorder(mask, n2r=True)

    if dtype is not None:
        mask = np.asarray(mask, dtype)
    elif apodization > 0:
        # Convert from big-endian float32 (numpy dtype '>f4') to native float32
        mask = np.asarray(mask, np.float32)
        
    return mask


def download():
    r"""Downloads Planck galmasks.
    
    Can be called from command line: ``python -m kszx download_planck``."""

    for apodization in [0,2,5]:
        _hfi_galmask_filename(apodization, download=True)
    

####################################################################################################


def _hfi_galmask_filename(apodization, download=False, dlfunc=None):
    """
    Allowed apodization values: 0, 2, 5 (degrees)

    Note that we use Planck release 2, since release 3 doesn't seem to have HFI
    foreground masks (it does have other masks).
    
    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """

    assert apodization in [0, 2, 5]    
    relpath = f'release_2/ancillary-data/masks/HFI_Mask_GalPlane-apo{apodization}_2048_R2.00.fits'
    
    planck_base_dir = os.path.join(io_utils.get_data_dir(), 'planck')
    abspath = os.path.join(planck_base_dir, relpath)

    if io_utils.do_download(abspath, download, dlfunc):
        url = f'https://irsa.ipac.caltech.edu/data/Planck/{relpath}'
        io_utils.wget(abspath, url)
    
    return abspath
