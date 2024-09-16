"""Not much here for now -- just some code to download DR10 randoms. More to come later."""

import os
from . import io_utils


def download_randoms(dr):
    for i in range(20):
        _desils_path(f'randoms/randoms-1-{i}.fits', dr, download=True)


def download_south_randoms(dr):
    for i in range(20):
        _desils_path(f'south/randoms/randoms-south-1-{i}.fits', dr, download=True)

        
def _desils_path(relpath, dr, download=False):
    """Example: _desils_path('randoms/randoms-1-0.fits', dr=10).
    Intended to be called through wrapper, e.g. _random_filename()."""
    
    assert dr == 10    # placeholder for future expansion
    relpath = os.path.join('dr10', relpath)

    desils_base_dir = io_utils.get_data_dir('desils')
    abspath = os.path.join(desils_base_dir, relpath)

    if download and not os.path.exists(abspath):
        url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
    
    return abspath
    
