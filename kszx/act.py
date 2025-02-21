"""
The ``kszx.act`` module contains functions for downloading/parsing ACT data products.

References: 
  - https://lambda.gsfc.nasa.gov/product/act/index.html
  - https://lambda.gsfc.nasa.gov/product/act/actadv_prod_table.html (DR5)
  - https://portal.nersc.gov/project/act (cluster masks from https://arxiv.org/abs/2307.01258)
"""

import os
import zipfile
import numpy as np
import pixell.enmap

from . import io_utils


def read_cmb(freq, dr, *, night=False, download=False):
    r"""Returns a pixell map. We currently only support DR5 act_planck_*_srcfree maps.

    Function args:
    
      - ``freq`` (integer): either 90, 150, or 220.
      - ``dr`` (integer): currently, only ``dr=5`` is supported.
      - ``night`` (boolean): either True (for night) or False (for daynight).
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.
    """
    
    filename = _cmb_filename(freq, dr, night=night, download=download)
    return _read_map(filename)


def read_ivar(freq, dr, *, night=False, download=False):
    r"""Returns a pixell map. We currently only support DR5 act_planck_daynight_srcfree maps.

    Function args:
    
      - ``freq`` (integer): either 90, 150, or 220.
      - ``dr`` (integer): currently, only ``dr=5`` is supported.
      - ``night`` (boolean): either True (for night) or False (for daynight).
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.
    """
    
    filename = _ivar_filename(freq, dr, night=night, download=download)
    return _read_map(filename)


def read_beam(freq, dr, lmax=None, *, night=False, download=False):
    r"""Returns a 1-d numpy array of length (lmax+1). We currently only support DR5 daynight beams.

    Function args:
    
      - ``freq`` (integer): either 90, 150, or 220.
      - ``dr`` (integer): currently, only ``dr=5`` is supported.
      - ``lmax`` (integer): if None, then a large lmax will be used.
      - ``night`` (boolean): either True (for night) or False (for daynight).
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.
    """

    filename = _beam_filename(freq, dr, night=night, download=download)
        
    print(f'Reading {filename}\n', end='')
    
    a = np.loadtxt(filename)
    assert (a.ndim==2) and (a.shape[1]==2)
    assert np.all(a[:,0] == np.arange(len(a)))

    a = a[:,1]
        
    if lmax is not None:
        assert lmax < len(a)
        a = a[:(lmax+1)]
        
    return a


def read_cluster_mask(download=False):
    r"""Returns the cluster mask from https://arxiv.org/abs/2307.01258 as a pixell map."""
    filename = _cluster_mask_filename(download)
    return pixell.enmap.read_map(filename)


def read_nilc_wide_mask(download=False):
    r"""Returns the Galactic mask from https://arxiv.org/abs/2307.01258 as a pixell map."""
    filename = _nilc_wide_mask_filename(download)
    return pixell.enmap.read_map(filename)


def download(dr, freq_list=None, night=False, cmb=True, ivar=True, beams=True):
    r"""Downloads ACT data products (cmb, ivar, beam) for a given survey.
        
    Can be called from command line: ``python -m kszx download_act``."""

    if freq_list is None:
        freq_list = [ 90, 150, 220 ]

    for freq in freq_list:
        _cmb_filename(freq, dr, night=night, download=cmb)
        _ivar_filename(freq, dr, night=night, download=ivar)
        _beam_filename(freq, dr, night=night, download=beams)


####################################################################################################


def _act_path(relpath, dr, download=False, is_aux=False):
    """Intended to be called through wrapper such as _cmb_filename(), _ivar_filename(), etc.

    For files which are downloaded here: https://lambda.gsfc.nasa.gov/data/suborbital/ACT
    If is_aux=True, then the file is contained in 'act_dr5.01_auxilliary.zip' (only matters if download=True)
    """

    assert dr == 5    # currently, only support DR5
    act_base_dir = os.path.join(io_utils.get_data_dir(), 'act', 'dr5.01')
    abspath = os.path.join(act_base_dir, relpath)

    if (not download) or os.path.exists(abspath):
        return abspath

    if not is_aux:
        # Typical case: file is not part of act_dr5.01_auxilliary.zip.
        url = f'https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
        return abspath

    # Special case: file is part of act_dr5.01_auxilliary.zip.
    # Download the zipfile (by calling _act_path() recursively) and unpack.

    zip_filename = _act_path('act_dr5.01_auxilliary.zip', dr=5, download=True, is_aux=False)
    print(f'Unzipping {zip_filename}')
    
    with zipfile.ZipFile(zip_filename, 'r') as f:
        f.extractall(act_base_dir)

    if not os.path.exists(abspath):
        raise RuntimeError(f"File '{relpath}' not found in zip file '{zip_filename}'")
    
    return abspath


def _act_nersc_path(relpath, download=False):
    """Intended to be called through wrapper such as _cluster_mask_filename().
    For files which are downloaded here: https://portal.nersc.gov/project/act
    """

    act_base_dir = os.path.join(io_utils.get_data_dir(), 'act')
    abspath = os.path.join(act_base_dir, relpath)

    if download and not os.path.exists(abspath):
        url = f'https://portal.nersc.gov/project/act/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading

    return abspath
    

def _cmb_filename(freq, dr, night=False, download=False):
    assert dr == 5   # currently, only support DR5
    daynight = 'night' if night else 'daynight'
    return _act_path(f'act_planck_dr5.01_s08s18_AA_f{freq:03d}_{daynight}_map_srcfree.fits', dr, download)

def _ivar_filename(freq, dr, night=False, download=False):
    assert dr == 5   # currently, only support DR5
    daynight = 'night' if night else 'daynight'
    return _act_path(f'act_planck_dr5.01_s08s18_AA_f{freq:03d}_{daynight}_ivar.fits', dr, download)

def _beam_filename(freq, dr, night=False, download=False):
    assert dr == 5   # currently, only support DR5
    daynight = 'night' if night else 'daynight'
    return _act_path(f'beams/act_planck_dr5.01_s08s18_f{freq:03d}_{daynight}_beam.txt', dr, download, is_aux=True)

def _cluster_mask_filename(download=False):
    return _act_nersc_path('dr6_nilc/ymaps_20230220/masks/cluster_mask.fits', download)

def _nilc_wide_mask_filename(download=False):
    return _act_nersc_path('dr6_nilc/ymaps_20230220/masks/wide_mask_GAL070_apod_1.50_deg_wExtended.fits', download)


def _read_map(filename):
    """Called by read_cmb() and read_ivar()."""
    
    print(f'Reading {filename}\n', end='')

    # FIXME is there a way to avoid reading TQU, if we only want T?
    m = pixell.enmap.read_map(filename)
    assert m.ndim == 3
    
    # Use of .copy() saves memory, by keeping T data while dropping reference to TQU data.
    return m[0].copy()
