import os
import zipfile
import numpy as np
import pixell.enmap

from . import io_utils


def read_cmb(freq, dr, download=False):
    """Returns a pixell map. We currently only support DR5 act_planck_daynight_srcfree maps."""
    
    filename = _cmb_filename(freq, dr, download)
    return _read_map(filename)


def read_ivar(freq, dr, download=False):
    """Returns a pixell map. We currently only support DR5 act_planck_daynight_srcfree maps."""
    
    filename = _ivar_filename(freq, dr, download)
    return _read_map(filename)


def read_beam(freq, dr, lmax=None, download=False):
    """Returns a 1-d numpy array of length (lmax+1). We currently only support DR5 daynight beams."""

    filename = _beam_filename(freq, dr, download)
        
    print(f'Reading {filename}')
            
    a = np.loadtxt(filename)
    assert (a.ndim==2) and (a.shape[1]==2)
    assert np.all(a[:,0] == np.arange(len(a)))

    a = a[:,1]
        
    if lmax is not None:
        assert lmax < len(a)
        a = a[:(lmax+1)]
        
    return a


def read_cluster_mask(download=False):
    filename = _cluster_mask_filename(download)
    return pixell.enmap.read_map(filename)


def read_nilc_wide_mask(download=False):
    filename = _nilc_wide_mask_filename(download)
    return pixell.enmap.read_map(filename)


def download(dr, freq_list=None, cmb=True, ivar=True, beams=True):
    if freq_list is None:
        freq_list = [ 90, 150, 220 ]

    for freq in freq_list:
        _cmb_filename(freq, dr, download=cmb)
        _ivar_filename(freq, dr, download=ivar)
        _beam_filename(freq, dr, download=beams)


####################################################################################################


def _act_path(relpath, dr, download=False, is_aux=False):
    """Intended to be called through wrapper such as _cmb_filename(), _ivar_filename(), etc.

    For files which are downloaded here: https://lambda.gsfc.nasa.gov/data/suborbital/ACT
    If is_aux=True, then the file is contained in 'act_dr5.01_auxilliary.zip' (only matters if download=True)
    """

    assert dr == 5    # currently, only support DR5
    act_base_dir = io_utils.get_data_dir('act')
    abspath = os.path.join(act_base_dir, 'dr5.01', relpath)

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
        raise RuntimeError(f"File '{relname}' not found in zip file '{zip_filename}'")
    
    return abspath


def _act_nersc_path(relpath, download=False):
    """Intended to be called through wrapper such as _cluster_mask_filename().
    For files which are downloaded here: https://portal.nersc.gov/project/act
    """

    act_base_dir = io_utils.get_data_dir('act')
    abspath = os.path.join(act_base_dir, relpath)

    if download and not os.path.exists(abspath):
        url = f'https://portal.nersc.gov/project/act/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading

    return abspath
    

def _cmb_filename(freq, dr, download=False):
    assert dr == 5   # currently, only support DR5
    return _act_path(f'act_planck_dr5.01_s08s18_AA_f{freq:03d}_daynight_map_srcfree.fits', dr, download)

def _ivar_filename(freq, dr, download=False):
    assert dr == 5   # currently, only support DR5
    return _act_path(f'act_planck_dr5.01_s08s18_AA_f{freq:03d}_daynight_ivar.fits', dr, download)

def _beam_filename(freq, dr, download=False):
    assert dr == 5   # currently, only support DR5
    return _act_path(f'beams/act_planck_dr5.01_s08s18_f{freq:03d}_daynight_beam.txt', dr, download, is_aux=True)

def _cluster_mask_filename(download=False):
    return _act_nersc_path('dr6_nilc/ymaps_20230220/masks/cluster_mask.fits', download)

def _nilc_wide_mask_filename(download=False):
    return _act_nersc_path('dr6_nilc/ymaps_20230220/masks/wide_mask_GAL070_apod_1.50_deg_wExtended.fits', download)


def _read_map(filename):
    """Called by read_cmb() and read_ivar()."""
    
    print(f'Reading {filename}')

    # FIXME is there a way to avoid reading TQU, if we only want T?
    m = pixell.enmap.read_map(filename)
    assert m.ndim == 3
    
    # Use of .copy() saves memory, by keeping T data while dropping reference to TQU data.
    return m[0].copy()
