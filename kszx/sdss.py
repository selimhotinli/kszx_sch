import os
import gzip
import shutil

from . import io_utils


def read_galaxies(survey, download=False):
    pass


def read_randoms(survey, download=False):
    pass


def download(survey):
    _galaxy_filename(survey, download=True)
    _random_filenames(survey, download=True)


####################################################################################################


def _check_survey(survey):
    """Check that 'survey' is valid, and return its standard capitalization."""
    
    survey_list = [ 'CMASS_North', 'CMASS_South', 'LOWZ_North', 'LOWZ_South',
                    'CMASSLOWZTOT_North', 'CMASSLOWZTOT_South', 'CLASSLOWZE2_North',
                    'CMASSLOWZE3_North' ]
                    
    for s in survey_list:
        if survey.upper() == s.upper():
            return s

    raise RuntimeError(f"SDSS survey '{survey}' not recognized")
        
    
def _sdss_path(relpath, download=False, gz=False):
    """Intended to be called through wrapper such as _galaxy_filename(), _random_filenames(), etc.
    If gz=True, then the file is gzipped on the SDSS website (only matters if download=True).
    """
    
    sdss_base_dir = io_utils.get_data_dir('sdss')
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
