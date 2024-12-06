"""
The ``kszx.sdss`` module contains functions for downloading/parsing SDSS data products.

References:
   - https://data.sdss.org/sas/dr12/boss/lss/
   - https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/
"""

import os
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

    Reminder: SDSS galaxies are usually weighted by ``(wzf + wcp − 1) * wsys * wkp``.

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
     wfkp                 # FKP weight

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


def read_mock(survey, mock_type, ix, download=False):
    r"""Reads SDSS mock galaxy catalog, and returns a Catalog object.

    Note that the qpm mocks are already restricted to an appropriate redshift range,
    i.e. there should be no need to call :func:`Catalog.apply_redshift_cut()` (unlike
    the "core" SDSS galaxy/random catalogs).

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South'.

      - ``mock_type`` (str): currently only 'qpm' is supported

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z,    # sky location, redshift
     wfkp,                  # FKP weight
     wveto                  # systematic weights

    Example usage::

     # kszx.sdss.read_mock() returns a kszx.Catalog.
     mcat = kszx.sdss.read_mock('CMASS_North', 'qpm', 0)   # index can be 0 <= ix < 1000
     # mcat.apply_redshift_cut(0.43, 0.7) not needed!
    """

    filename = _mock_filename(survey, mock_type, ix, download=download)

    if mock_type.lower() == 'qpm':
        # https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/dr11_qpm_mocks/mock_galaxy_DRX_SAMPLE_NS_QPM_IDNUMBER.html
        return Catalog.from_text_file(filename, col_names=['ra_deg','dec_deg','z','wfkp','wveto'])
    else:
        raise RuntimeError(f"kszx.sdss: {mock_type=} was not recognized (only 'qpm' is currently supported)")


def read_mock_randoms(survey, mock_type, download=True):
    r"""Reads SDSS mock random catalog, and returns a Catalog object.

    Note that the qpm mocks are already restricted to an appropriate redshift range,
    i.e. there should be no need to call :func:`Catalog.apply_redshift_cut()` (unlike
    the "core" SDSS galaxy/random catalogs).

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South'.

      - ``mock_type`` (str): currently only 'qpm' is supported.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z   # sky location, redshift
     wfkp                 # FKP weight

    Example usage::

     # kszx.sdss.read_mock_randoms() returns a kszx.Catalog.
     mrcat = kszx.sdss.read_mock_randoms('CMASS_North', 'qpm')
     # mrcat.apply_redshift_cut(0.43, 0.7) not needed!
    """

    if mock_type.lower() == 'qpm':
        # https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/dr11_qpm_mocks/mock_random_DRX_SAMPLE_NS_QPM_NxV.html
        filenames = _mock_random_filenames(survey, mock_type, download=download)
        catalogs = [ Catalog.from_text_file(f, col_names=['ra_deg','dec_deg','z','wfkp']) for f in filenames ]
        return Catalog.concatenate(catalogs, name=f'{survey} qpm mock randoms', destructive=True)
    else:
        raise RuntimeError(f"kszx.sdss: {mock_type=} was not recognized (only 'qpm' is currently supported)")
    

def download(survey, mask=False, qpm=False):
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
    _mock_filename(survey, 'qpm', 999, download=qpm)   # will also download mocks 0, ..., 998
    _mock_random_filename(survey, 'qpm', download=qpm)


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
        #   https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/randomN_DRX_SAMPLE_NS.html
        
        catalog.add_column('ra_deg', f[1].read('RA'))
        catalog.add_column('dec_deg', f[1].read('DEC'))
        catalog.add_column('z', f[1].read('Z'))
        catalog.add_column('wfkp', f[1].read('WEIGHT_FKP'))

        # catalog.add_column('ipoly', f[1].read('IPOLY'))
        # catalog.add_column('isect', f[1].read('ISECT'))

        if not is_randcat:
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


def _check_survey(survey, abridged=False):
    """Check that 'survey' is valid, and return its standard capitalization."""
    
    survey_list = [ 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South' ]

    if not abridged:
        survey_list += [ 'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South',
                         'CLASSLOWZE2_North', 'CMASSLOWZE3_North' ]
                    
    for s in survey_list:
        if survey.upper() == s.upper():
            return s

    raise RuntimeError(f"SDSS survey '{survey}' not recognized (must be one of: {survey_list})")
        
    
def _sdss_path(relpath, download=False, *, packed_relpath=None, gz=False):
    """Intended to be called through wrapper such as _galaxy_filename(), _random_filenames(), etc.

    If 'packed_relpath' is specified, it should be a file ending in '.gz', '.tgz', or '.tar.gz'
    which generates 'relpath' after unpacking.

    Setting gz=True is shorthand for packed_relpath = "{relpath}.gz".

    Example: 

         _sdss_path(relpath = 'qpm_mocks/mock_galaxy_DR12_LOWZ_S_QPM_0137.rdzw',
                    packed_relpath = 'qpm_mocks/mock_galaxy_DR12_LOWZ_S_QPM_allmocks.tar.gz',
                    download = True)

    Does the following:

       - If the following file is found, then return:
          {kszx_data_dir}/sdss/DR12v5/qpm_mocks/mock_galaxy_DR12_LOWZ_S_QPM_0137.rdzw

       - If the following file is found, then unpack and return:
          {kszx_data_dir}/sdss/DR12v5/qpm_mocks/mock_galaxy_DR12_LOWZ_S_QPM_allmocks.tar.gz

       - Download and decompress:
           https://data.sdss.org/sas/dr12/boss/lss/qpm_mocks/mock_galaxy_DR12_LOWZ_S_QPM_allmocks.tar.gz
    """

    if gz and (packed_relpath is not None):
        raise RuntimeError("kszx.sdss internal error: can't specify both 'packed_relpath' and 'gz'")
    
    sdss_base_dir = os.path.join(io_utils.get_data_dir(), 'sdss')
    abspath = os.path.join(sdss_base_dir, 'DR12v5', relpath)
        
    if (not download) or os.path.exists(abspath):
        return abspath

    if gz:
        packed_relpath = relpath + '.gz'

    if not packed_relpath:
        # Case 1: No tar/gz involved, just download file.
        url = f'https://data.sdss.org/sas/dr12/boss/lss/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(abspath) after downloading
    else:
        # Case 2: Download tar/gz file by calling _sdss_path() recursively, then unpack.
        packed_abspath = _sdss_path(packed_relpath, download=True)
        io_utils.unpack(packed_abspath, abspath)  # calls assert os.path.exists(abspath) after unpacking
    
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


def _mock_filename(survey, mock_type, ix, download=False):
    _check_survey(survey, abridged=True)
    
    if mock_type.lower() != 'qpm':
        raise RuntimeError(f"kszx.sdss: {mock_type=} was not recognized (only 'qpm' is currently supported)")

    # NOTE: 'ix' argument is a zero-based index 0 <= ix < 1000, but the
    # QPM files use a one-based index 1 <= ix <= 1000, so we add 1.
    
    return _sdss_path(
        relpath = f'qpm_mocks/mock_galaxy_DR12_{survey[:-4].upper()}_QPM_{(ix+1):04d}.rdzw',
        packed_relpath = f'qpm_mocks/mock_galaxy_DR12_{survey[:-4].upper()}_QPM_allmocks.tar.gz',
        download = download
    )


def _mock_random_filenames(survey, mock_type, download=False):
    _check_survey(survey, abridged=True)
    
    if mock_type.lower() != 'qpm':
        raise RuntimeError(f"kszx.sdss: {mock_type=} was not recognized (only 'qpm' is currently supported)")

    return [ _sdss_path(f'qpm_mocks/mock_random_DR12_{survey[:-4].upper()}_50x{n}.rdzw', download=True, gz=True) for n in [1,2] ]
