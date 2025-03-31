"""
The ``kszx.desi`` module contains functions for downloading/parsing DESI data products.

References:
   - https://data.desi.lbl.gov/doc/releases/dr1/
   - https://data.desi.lbl.gov/doc/organization/
   - https://desidatamodel.readthedocs.io/en/latest/
   - https://github.com/desihub/tutorials
   - https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/
"""


import os
import fitsio

from . import Catalog
from . import io_utils


def read_galaxies(survey, dr, download=False):
    r"""Reads DESI galaxy catalog, and returns a Catalog object.

    Function arguments:

      - ``survey`` (str): one of the following:
    
           'LRG_NGC', 'LRG_SGC',
           'QSO_NGC', 'QSO_SGC'
           'BGS_ANY_NGC', 'BGS_ANY_SGC',
           'BGS_BRIGHT_NGC', 'BGS_BRIGHT_SGC',
           'BGS_BRIGHT-21.5_NGC', 'BGS_BRIGHT-21.5_SGC',
           'LRG+ELG_LOPnotqso_NGC', 'LRG+ELG_LOPnotqso_SGC'

      - ``dr`` (int): currently only dr=1 is supported

    FIXME: finish docstring (use sdss.read_galaxies docstring as a model)
    """

    filename = _galaxy_filename(survey, dr, download, dlfunc='kszx.desi.read_galaxies')
    gcat = read_fits_catalog(filename, name=survey)
    return gcat


def read_randoms(survey, dr=12, download=False, nfiles=None):
    r"""Reads DESI random catalog, and returns a Catalog object.
    
    Function arguments:

      - ``survey`` (str): one of the following:
    
           'LRG_NGC', 'LRG_SGC',
           'QSO_NGC', 'QSO_SGC'
           'BGS_ANY_NGC', 'BGS_ANY_SGC',
           'BGS_BRIGHT_NGC', 'BGS_BRIGHT_SGC',
           'BGS_BRIGHT-21.5_NGC', 'BGS_BRIGHT-21.5_SGC',
           'LRG+ELG_LOPnotqso_NGC', 'LRG+ELG_LOPnotqso_SGC'

      - ``dr`` (int): currently only dr=1 is supported

    FIXME: finish docstring (use sdss.read_randoms docstring as a model)
    """
    
    filenames = _random_filenames(survey, dr, download, nfiles, dlfunc='kszx.desi.read_randoms')
    catalogs = [ read_fits_catalog(f) for f in filenames ]
    rcat = Catalog.concatenate(catalogs, name=f'{survey} randoms', destructive=True)
    return rcat


def download(survey, dr):
    """FIXME: write docstring."""
    
    _galaxy_filename(survey, dr, download=True)
    _random_filenames(survey, dr, download=True)



####################################################################################################


def read_fits_catalog(filename, name=None):
    """FIXME write docstring."""

    print(f'Reading {filename}\n', end='')
    catalog = Catalog(name=name, filename=filename)

    with fitsio.FITS(filename) as f:
        
        # Empirically, galaxy/random FITS files contain the following columns:
        #
        #     RA                  f8  
        #     DEC                 f8  
        #     NTILE               i8  
        #     PHOTSYS             S1  
        #     FRAC_TLOBS_TILES    f8
        #     Z                   f8  
        #     WEIGHT              f8  
        #     WEIGHT_SYS          f8  
        #     WEIGHT_COMP         f8  
        #     WEIGHT_ZFAIL        f8  
        #     TARGETID_DATA       i8  
        #     NX                  f8  
        #     WEIGHT_FKP          f8  
        
        catalog.add_column('ra_deg', f[1].read('RA'))
        catalog.add_column('dec_deg', f[1].read('DEC'))
        catalog.add_column('z', f[1].read('Z'))
        catalog.add_column('weight', f[1].read('WEIGHT'))
        catalog.add_column('wfkp', f[1].read('WEIGHT_FKP'))
        catalog.add_column('wcp', f[1].read('WEIGHT_COMP'))   # FIXME is this the same as SDSS WEIGHT_CP?
        catalog.add_column('wzf', f[1].read('WEIGHT_ZFAIL'))  # FIXME is this the same as SDSS WEIGHT_CP?
        catalog.add_column('wsys', f[1].read('WEIGHT_SYS'))

    catalog._announce_file_read()
    return catalog

    
def _check_survey(survey, dr):
    """Check that 'survey' is valid, and return its standard capitalization."""
    
    survey_list = [ 
        'LRG_NGC', 'LRG_SGC',
        'QSO_NGC', 'QSO_SGC',
        'BGS_ANY_NGC', 'BGS_ANY_SGC',
        'BGS_BRIGHT_NGC', 'BGS_BRIGHT_SGC',
        'BGS_BRIGHT-21.5_NGC', 'BGS_BRIGHT-21.5_SGC',
        'LRG+ELG_LOPnotqso_NGC', 'LRG+ELG_LOPnotqso_SGC'
    ]
    
    if dr != 1:
        raise RuntimeError(f"DESI {dr=} not supported (currently only dr=1 is supported)")
    
    for s in survey_list:
        if survey.upper() == s.upper():
            return s

    raise RuntimeError(f"DESI survey '{survey}' not recognized (must be one of: {survey_list})")


def _desi_path(relpath, dr, download, dlfunc=None):
    """Intended to be called through wrapper: _galaxy_filename(), _random_filenames(), etc.

    Local filename is:
       {desi_base_dir}/dr1/{relpath}
    
    Remote URL is:
       https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/{relpath}

    Note that 'relpath' will begin with v1.5/.
    
    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """

    assert dr == 1   # for now
    desi_base_dir = os.path.join(io_utils.get_data_dir(), 'desi')
    abspath = os.path.join(desi_base_dir, f'dr{dr}', relpath)

    if io_utils.do_download(abspath, download, dlfunc):
        url = f'https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/{relpath}'
        io_utils.wget(abspath, url)  # creates directories if needed, asserts os.path.exists(abspath) after download

    return abspath
    

def _galaxy_filename(survey, dr, download=False, dlfunc=None):
    s = _check_survey(survey, dr)
    return _desi_path(f'v1.5/{s}_clustering.dat.fits', dr, download, dlfunc)


def _random_filenames(survey, dr, download=False, nfiles=None, dlfunc=None):
    if nfiles is None:
        nfiles = 18  # default
        
    s = _check_survey(survey, dr)
    return [ _desi_path(f'v1.5/{s}_{i}_clustering.ran.fits', dr, download, dlfunc) for i in range(nfiles) ]
