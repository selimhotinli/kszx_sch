"""
The ``kszx.sdss`` module contains functions for downloading/parsing SDSS data products.

References:
   - https://data.sdss.org/sas/dr12/boss/lss/
   - https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/
"""

import os
import gzip
import shutil
import fitsio

from . import Catalog
from . import io_utils


def read_galaxies(survey, download=False):
    r"""Reads SDSS galaxy catalog, and returns a Catalog object.
    
    After calling this function, you'll probably want to call :func:`Catalog.apply_redshift_cut()`
    to retrict to an appropriate redshift range.

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z,    # sky location, redshift
     wfkp,                  # FKP weight
     wcp, wzf, wsys,        # systematic weights
     cboss,                 # completeness
     id                     # unique identifier

    Example usage::

     # kszx.sdss.read_galaxies() returns a kszx.Catalog.
     gcat = kszx.sdss.read_galaxies('CMASS_North')
     gcat.apply_redshift_cut(0.43, 0.7)
    """
    
    filename = _galaxy_filename(survey, download)
    return read_fits_catalog(filename, is_randcat=False)


def read_randoms(survey, download=False):
    r"""Reads SDSS random catalog, and returns a Catalog object.

    After calling this function, you'll probably want to call :func:`Catalog.apply_redshift_cut()`
    to retrict to an appropriate redshift range.

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z   # sky location, redshift

    Example usage::

     # kszx.sdss.read_randoms() returns a kszx.Catalog.
     rcat = kszx.sdss.read_randoms('CMASS_North')
     rcat.apply_redshift_cut(0.43, 0.7)
    """
    
    filenames = _random_filenames(survey, download)
    catalogs = [ read_fits_catalog(f, is_randcat=True) for f in filenames ]
    return Catalog.concatenate(catalogs, name=f'{survey} randoms', destructive=True)


def read_mask(survey, download=False):
    r"""Reads SDSS mangle mask, and returns a pymangle object. Rarely needed!

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a pymangle object that can be evaluated with mask.weight(ra_deg, dec_deg).
    The return value is either 0, or close to 1 (e.g. 0.98).

    Note: requires pymangle! (Not a dependency by default.)    
    """
    import pymangle
    filename = _mask_filename(survey, download)
    print(f'Reading {filename}')
    return pymangle.Mangle(filename)


def download(survey, mask=False):
    r"""Downloads SDSS data products (galaxies, randoms, and optionally mangle mask) for a given survey.

    Can be called from command line: ``python -m kszx download_sdss``.

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``mask`` (boolean): if True, then mangle mask will be downloaded (in addition to 
        galaxies and randoms).
    """
    
    _galaxy_filename(survey, download=True)
    _random_filenames(survey, download=True)
    _mask_filename(survey, download=mask)


####################################################################################################


def read_fits_catalog(filename, is_randcat, name=None, extra_columns=[]):
    r"""Reads FITS file in SDSS catalog format. 

    Intended as a helper for read_galaxies() or read_randoms(), but may be useful elsewhere.
    
    Function arguments:

      - ``filename`` (string): should end in ``.fits``.
      - ``is_randcat`` (boolean): True for a random catalog, False for a galaxy catalog.
        (Determines which column names are expected).
      - ``name`` (str, optional): name of Catalog, passed to :class:`kszx.Catalog` constructor.
      - ``extra_columns`` (list of string): extra columns to read (if any).

    Returns a :class:`kszx.Catalog` object.
    """

    print(f'Reading {filename}')
    catalog = Catalog(name=name, filename=filename)

    with fitsio.FITS(filename) as f:
        
        # For a list of all fields, see
        #   https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/galaxy_DRX_SAMPLE_NS.html
        #   https://data.sdss3.org/datamodel/files/BOSS_LSS_REDUX/galaxy_DR11v1_SAMPLE_NS.html
        #   https://data.sdss3.org/datamodel/files/BOSS_LSS_REDUX/randomN_DR10v8_SAMPLE_NS.html
        
        catalog.add_column('ra_deg', f[1].read('RA'))
        catalog.add_column('dec_deg', f[1].read('DEC'))
        catalog.add_column('z', f[1].read('Z'))

        # catalog.add_column('ipoly', f[1].read('IPOLY'))
        # catalog.add_column('isect', f[1].read('ISECT'))

        if not is_randcat:
            catalog.add_column('wfkp', f[1].read('WEIGHT_FKP'))
            catalog.add_column('wcp', f[1].read('WEIGHT_CP'))
            catalog.add_column('wzf', f[1].read('WEIGHT_NOZ'))
            catalog.add_column('wsys', f[1].read('WEIGHT_SYSTOT'))
            catalog.add_column('cboss', f[1].read('COMP'))
            catalog.add_column('id', f[1].read('ID'))

        for col_name in extra_columns:
            catalog.add_column(col_name.lower(), f[1].read(col_name.upper()))

    catalog._announce_file_read()
    return catalog


####################################################################################################


def _check_survey(survey):
    """Check that 'survey' is valid, and return its standard capitalization."""
    
    survey_list = [ 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
                    'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North',
                    'CMASSLOWZE3_North' ]
                    
    for s in survey_list:
        if survey.upper() == s.upper():
            return s

    raise RuntimeError(f"SDSS survey '{survey}' not recognized (must be one of: {survey_list})")
        
    
def _sdss_path(relpath, download=False, gz=False):
    """Intended to be called through wrapper such as _galaxy_filename(), _random_filenames(), etc.
    If gz=True, then the file is gzipped on the SDSS website (only matters if download=True).
    """
    
    sdss_base_dir = os.path.join(io_utils.get_data_dir(), 'sdss')
    abspath = os.path.join(sdss_base_dir, 'DR12v5', relpath)

    if (not download) or os.path.exists(abspath):
        return abspath

    if not gz:
        url = f'https://data.sdss.org/sas/dr12/boss/lss/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
        return abspath

    # Case gz=True. Download gzipped file (by calling _sdss_path() recursively) and uncompress.
    
    gzpath = _sdss_path(f'{relpath}.gz', download=True, gz=False)
    print(f'Uncompressing {gzpath}')

    with gzip.open(gzpath, 'rb') as f_in:
        with open(abspath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return abspath


def _galaxy_filename(survey, download=False):
    s = _check_survey(survey)
    return _sdss_path(f'galaxy_DR12v5_{s}.fits', download, gz=True)


def _random_filenames(survey, download=False):
    s = _check_survey(survey)
    return [ _sdss_path(f'random{n}_DR12v5_{s}.fits', download, gz=True) for n in [0,1] ]


def _mask_filename(survey, download=False):
    s = _check_survey(survey)
    return _sdss_path(f'mask_DR12v5_{s}.ply', download)
