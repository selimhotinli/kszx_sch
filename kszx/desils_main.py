"""Not much here for now -- just some code to download DR10 randoms. More to come later."""

import os
from . import io_utils


def download_randoms(dr):
    for i in range(20):
        _desils_path(f'randoms/randoms-1-{i}.fits', dr, download=True)


def download_south_randoms(dr):
    for i in range(20):
        _desils_path(f'south/randoms/randoms-south-1-{i}.fits', dr, download=True)

        
def _desils_path(relpath, dr, download=False, dlfunc=None):
    """Example: _desils_path('randoms/randoms-1-0.fits', dr=10).
    
    Intended to be called through wrapper, e.g. _random_filename().
    
    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """
    
    assert dr == 10    # placeholder for future expansion
    relpath = os.path.join('dr10', relpath)

    desils_base_dir = os.path.join(io_utils.get_data_dir(), 'desils')
    abspath = os.path.join(desils_base_dir, relpath)

    if io_utils.do_download(abspath, download, dlfunc):
        url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
    
    return abspath
    
