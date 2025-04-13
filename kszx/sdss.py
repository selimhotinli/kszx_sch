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


def read_galaxies(survey, dr=12, download=False):
    r"""Reads SDSS galaxy catalog, and returns a Catalog object.
    
    After calling this function, you'll probably want to call :func:`Catalog.apply_redshift_cut()`
    to retrict to an appropriate redshift range.

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``dr`` (integer): either 11 or 12.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z,    # sky location, redshift
     wfkp,                  # FKP weight
     wcp, wzf, wsys,        # systematic weights
     cboss,                 # completeness (denoted C_BOSS in papers and COMP in the FITS file)
     id                     # unique identifier

    Reminder: SDSS galaxies are usually weighted by ``(wzf + wcp − 1) * wsys * wkp``.

    Example usage::

     # kszx.sdss.read_galaxies() returns a kszx.Catalog.
     gcat = kszx.sdss.read_galaxies('CMASS_North')
     gcat.apply_redshift_cut(0.43, 0.7)
    """
    
    filename = _galaxy_filename(survey, dr, download, dlfunc='kszx.sdss.read_galaxies')
    gcat = read_fits_catalog(filename, is_randcat=False)
    print("Reminder: you probably want to call cat.apply_redshift_cut(zmin,zmax)"
          + " on the Catalog returned by kszx.sdss.read_galaxies())")
    return gcat


def read_randoms(survey, dr=12, download=False):
    r"""Reads SDSS random catalog, and returns a Catalog object.

    After calling this function, you'll probably want to call :func:`Catalog.apply_redshift_cut()`
    to retrict to an appropriate redshift range.

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``dr`` (integer): either 11 or 12.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z   # sky location, redshift
     wfkp                 # FKP weight

    Example usage::

     # kszx.sdss.read_randoms() returns a kszx.Catalog.
     rcat = kszx.sdss.read_randoms('CMASS_North')
     rcat.apply_redshift_cut(0.43, 0.7)
    """
    
    filenames = _random_filenames(survey, dr, download, dlfunc='kszx.sdss.read_randoms')
    catalogs = [ read_fits_catalog(f, is_randcat=True) for f in filenames ]
    rcat = Catalog.concatenate(catalogs, name=f'{survey} randoms', destructive=True)
    print("Reminder: you probably want to call cat.apply_redshift_cut(zmin,zmax)"
          + " on the Catalog returned by kszx.sdss.read_randoms())")
    return rcat


def read_mask(survey, dr=12, download=False):
    r"""Reads SDSS mangle mask, and returns a pymangle object. Rarely needed!

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``dr`` (integer): either 11 or 12.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a pymangle object that can be evaluated with mask.weight(ra_deg, dec_deg).
    The return value is either 0, or close to 1 (e.g. 0.98).

    Note: requires pymangle! (Not a dependency by default.)    
    """
    import pymangle
    filename = _mask_filename(survey, dr, download, dlfunc='kszx.sdss.read_mask')
    print(f'Reading {filename}\n', end='')
    return pymangle.Mangle(filename)


def read_mock(survey, mock_type, ix, dr=12, download=False):
    r"""Reads SDSS mock galaxy catalog, and returns a Catalog object.

    Note that the qpm mocks are already restricted to an appropriate redshift range,
    i.e. there should be no need to call :func:`Catalog.apply_redshift_cut()` (unlike
    the "core" SDSS galaxy/random catalogs).

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South'.

      - ``mock_type`` (str): currently 'qpm' (DR12 only) and 'pthalos' (DR11 only)
        are implemented.

      - ``ix`` (integer): index of mock, in the range ``0 <= ix < 1000``, except for
        CMASS PTHALOS where the range is ``0 <= ix < 600``.

      - ``dr`` (integer): either 11 or 12.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z,    # sky location, redshift
     wveto                  # systematic weights
     wfkp                   # FKP weight (available for QPM but not PTHALOS)
     cboss, wcp, wzf        # PTHALOS only: completeness/systematics weights
     ztrue                  # PTHALOS only: true redshift, removing peculiar velocities

    Example usage::

     # kszx.sdss.read_mock() returns a kszx.Catalog.
     mcat = kszx.sdss.read_mock('CMASS_North', 'qpm', 0)   # index can be 0 <= ix < 1000
     # mcat.apply_redshift_cut(0.43, 0.7) not needed!
    """

    filename = _mock_filename(survey, mock_type, ix, dr, download=download, dlfunc='kszx.sdss.read_mock')

    if mock_type.lower() == 'qpm':
        # https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/dr11_qpm_mocks/mock_galaxy_DRX_SAMPLE_NS_QPM_IDNUMBER.html
        return Catalog.from_text_file(filename, col_names=['ra_deg','dec_deg','z','wfkp','wveto'])
    elif mock_type.lower() == 'pthalos':
        # https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/dr11_pthalos_mocks/mock_galaxy_DRX_SAMPLE_NS_PTHALOS_IDNUMBER.html
        # Omit 'ipoly' and 'galaxy flag'
        return Catalog.from_text_file(filename, col_names=['ra_deg','dec_deg','z', None,'cboss','wcp','wzf','wveto','ztrue',None])
    else:
        raise RuntimeError(f"kszx: SDSS {mock_type=} was not recognized (currently support 'qpm' and 'pthalos')")


def read_mock_randoms(survey, mock_type, dr=12, download=False):
    r"""Reads SDSS mock random catalog, and returns a Catalog object.

    Note that the qpm mocks are already restricted to an appropriate redshift range,
    i.e. there should be no need to call :func:`Catalog.apply_redshift_cut()` (unlike
    the "core" SDSS galaxy/random catalogs).

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South'.

      - ``mock_type`` (str): currently 'qpm' (DR12 only) and 'pthalos' (DR11 only)
        are implemented.

      - ``dr`` (integer): either 11 or 12.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
     
     ra_deg, dec_deg, z   # sky location, redshift
     wfkp                 # FKP weight

    Example usage::

     # kszx.sdss.read_mock_randoms() returns a kszx.Catalog.
     mrcat = kszx.sdss.read_mock_randoms('CMASS_North', 'qpm')
     # mrcat.apply_redshift_cut(0.43, 0.7) not needed!
    """

    filenames = _mock_random_filenames(survey, mock_type, dr, download=download, dlfunc='kszx.sdss.read_mock_randoms')

    if mock_type.lower() == 'qpm':
        # https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/dr11_qpm_mocks/mock_random_DRX_SAMPLE_NS_QPM_NxV.html
        catalogs = [ Catalog.from_text_file(f, col_names=['ra_deg','dec_deg','z','wfkp']) for f in filenames ]
        return Catalog.concatenate(catalogs, name=f'{survey} {mock_type} mock randoms', destructive=True)
    elif mock_type.lower() == 'pthalos':
        # https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/dr11_pthalos_mocks/mock_random_DRX_SAMPLE_NS_PTHALOS_IDNUMBER.html
        catalogs = [ Catalog.from_text_file(f, col_names=['ra_deg','dec_deg','z',None,'cboss','wcp','wzf','wveto']) for f in filenames ]
        return Catalog.concatenate(catalogs, name=f'{survey} {mock_type} randoms', destructive=True)        
    else:
        raise RuntimeError(f"kszx: SDSS {mock_type=} was not recognized (currently support 'qpm' and 'pthalos')")
    

def download(survey, dr=12, mask=False, qpm=False, pthalos=False):
    r"""Downloads SDSS data products (galaxies, randoms, and optionally mocks + mangle mask) for a given survey.

    Can be called from command line: ``python -m kszx download_sdss``.

    Function arguments:

      - ``survey`` (str): either 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
        'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North', or 'CMASSLOWZE3_North'.

      - ``dr`` (integer): either 11 or 12.

      - ``mask`` (boolean): if True, then mangle mask will be downloaded (in addition to 
        galaxies and randoms).

      - ``qpm`` (boolean): if True, then QPM mocks will be downloaded (DR12 only).

      - ``pthalos`` (boolean): if True, then PTHALOS mocks will be downloaded (DR11 only).
    """
    
    _galaxy_filename(survey, dr, download=True)
    _random_filenames(survey, dr, download=True)
    _mask_filename(survey, dr, download=mask)

    for flag, mock_type in [ (qpm,'qpm'), (pthalos,'pthalos') ]:
        if not flag:
            continue
        for _ in _mock_random_filenames(survey, mock_type, dr, download=True):
            pass
        _mock_filename(survey, mock_type, 0, dr, download=True)  # will download all mocks


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

    print(f'Reading {filename}\n', end='')
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


def _dr_str(dr):
    dhash = { 11: 'DR11v1', 12: 'DR12v5' }
    
    if dr not in dhash:
        raise RuntimeError(f"SDSS {dr=} not supported (currently support DR11, DR12)")

    return dhash[dr]


def _check_survey(survey, dr, abridged=False):
    """Checks 'survey' for validity, and returns its standard capitalization."""

    if dr not in [11,12]:
        raise RuntimeError(f"SDSS {dr=} not supported (currently support DR11, DR12)")
    
    survey_list = [ 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South' ]

    if (dr==12) and (not abridged):
        survey_list += [ 'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South',
                         'CLASSLOWZE2_North', 'CMASSLOWZE3_North' ]
                    
    for s in survey_list:
        if survey.upper() == s.upper():
            return s

    raise RuntimeError(f"SDSS survey '{survey}' not recognized (must be one of: {survey_list})")
        
    
def _sdss_path(relpath, dr, download, *, packed_relpath=None, gz=False, dlfunc=None):
    """Intended to be called through wrapper such as _galaxy_filename(), _random_filenames(), etc.

    If 'packed_relpath' is specified, it should be a file ending in '.gz', '.tgz', or '.tar.gz'
    which generates 'relpath' after unpacking.

    Setting gz=True is shorthand for packed_relpath = "{relpath}.gz".

    The 'dr' argument is only used if download=True, to construct the URL.
    
    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).

    Example: 

         _sdss_path(relpath = 'qpm_mocks/mock_galaxy_DR12_LOWZ_S_QPM_0137.rdzw',
                    packed_relpath = 'qpm_mocks/mock_galaxy_DR12_LOWZ_S_QPM_allmocks.tar.gz',
                    download = True, dr = 12)

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
    abspath = os.path.join(sdss_base_dir, _dr_str(dr), relpath)

    if not io_utils.do_download(abspath, download, dlfunc):
        return abspath

    if gz:
        packed_relpath = relpath + '.gz'

    if not packed_relpath:
        # Case 1: No tar/gz involved, just download file.
        url = f'https://data.sdss.org/sas/dr{dr}/boss/lss/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(abspath) after downloading
    else:
        # Case 2: Download tar/gz file by calling _sdss_path() recursively, then unpack.
        packed_abspath = _sdss_path(packed_relpath, dr, download=True)
        io_utils.unpack(packed_abspath, abspath)  # calls assert os.path.exists(abspath) after unpacking
    
    return abspath


def _galaxy_filename(survey, dr, download=False, dlfunc=None):
    d = _dr_str(dr)
    s = _check_survey(survey, dr)
    return _sdss_path(f'galaxy_{d}_{s}.fits', dr, download, gz=True, dlfunc=dlfunc)


def _random_filenames(survey, dr, download=False, dlfunc=None):
    d = _dr_str(dr)
    s = _check_survey(survey, dr)
    return [ _sdss_path(f'random{n}_{d}_{s}.fits', dr, download, gz=True, dlfunc=dlfunc) for n in [0,1] ]


def _mask_filename(survey, dr, download=False, dlfunc=None):
    d = _dr_str(dr)
    s = _check_survey(survey, dr)
    return _sdss_path(f'mask_{d}_{s}.ply', dr, download, dlfunc=dlfunc)


def _mock_filename(survey, mock_type, ix, dr, download=False, dlfunc=None):
    _check_survey(survey, dr, abridged=True)

    if (dr == 12) and (mock_type.lower() == 'qpm'):
        assert 0 <= ix < 1000
        
        # The 'ix' argument is a zero-based index 0 <= ix < 1000, but the
        # QPM files use a one-based index 1 <= ix <= 1000, so we add 1.
            
        return _sdss_path(
            relpath = f'qpm_mocks/mock_galaxy_DR{dr}_{survey[:-4].upper()}_QPM_{(ix+1):04d}.rdzw',
            packed_relpath = f'qpm_mocks/mock_galaxy_DR{dr}_{survey[:-4].upper()}_QPM_allmocks.tar.gz',
            download = download,
            dr = dr,
            dlfunc = dlfunc
        )

    elif (dr == 11) and (mock_type.lower() == 'pthalos'):
        assert 0 <= ix < 600
        
        # The 'ix' argument is a zero-based index 0 <= ix < 600, but the
        # pthalos mocks use an one-index 4001 <= ir <= 4600, so we add 4001 (!)

        return _sdss_path(
            relpath = f'dr{dr}_pthalos_mocks/mock_galaxy_DR{dr}_{survey[:-4].upper()}_PTHALOS_ir{(ix+4001)}.dat',
            packed_relpath = f'dr{dr}_pthalos_mocks/mock_galaxy_DR{dr}_{survey[:-4].upper()}_PTHALOS_allmocks.tar.gz',
            download = download,
            dr = dr,
            dlfunc = dlfunc
        )
    
    else:
        raise RuntimeError(f"kszx: SDSS {(mock_type,dr)=} was not recognized (currently support ('qpm',12) and ('pthalos',11))")


def _mock_random_filenames(survey, mock_type, dr, download=False, dlfunc=None):
    _check_survey(survey, dr, abridged=True)
    
    if (dr == 12) and (mock_type.lower() == 'qpm'):
        for n in [1,2]:
            yield _sdss_path(f'qpm_mocks/mock_random_DR{dr}_{survey[:-4].upper()}_50x{n}.rdzw', dr=dr, download=download, gz=True, dlfunc=dlfunc)

    elif (dr == 11) and (mock_type.lower() == 'pthalos'):
        nfiles = 600 if survey.lower().startswith('cmass') else 1000
        for n in range(4001, 4001+nfiles):
            yield _sdss_path(
                relpath = f'dr{dr}_pthalos_mocks/mock_random_DR{dr}_{survey[:-4].upper()}_PTHALOS_ir{n}.dat',
                packed_relpath = f'dr{dr}_pthalos_mocks/mock_random_DR{dr}_{survey[:-4].upper()}_PTHALOS_allmocks.tar.gz',
                download = download,
                dr = dr,
                dlfunc = dlfunc
            )
    
    else:
        raise RuntimeError(f"kszx: SDSS {(mock_type,dr)=} was not recognized (currently support ('qpm',12) and ('pthalos',11))")
