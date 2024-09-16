"""
Data products from https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/.

Note: these LRGs are DR9, not DR10!
"""

import os
import fitsio
import healpy
import functools
import numpy as np

from . import Catalog
from . import io_utils
    

def read_galaxies(extended, download=False):
    """Returns LRGs with no "quality cuts"."""

    # List of pairs (my_col_name, fits_col_name).
    cols = [ ('ra_deg','RA'), ('dec_deg','DEC'), ('z','Z_PHOT_MEDIAN') ]
    cols += [ ('ebv','EBV'), ('lrg_mask','lrg_mask') ]
    cols += [ (f'nobs_{x}', f'PIXEL_NOBS_{x.upper()}') for x in ['g','r','z'] ]
    cols += [ ('maskbits','MASKBITS') ]

    filename = _catalog_filename(extended, download=download)
    catalog = Catalog.from_fits(filename, cols)

    # zerr column (photo-z error)
    filename = _catalog_filename(extended, suffix='pz', download=download)
    ctmp = Catalog.from_fits(filename, [('zerr','Z_PHOT_STD')])
    catalog.add_column('zerr', ctmp.zerr)

    # stardens column
    m = read_stardens_map(download=download)
    nside = healpy.npix2nside(len(m))
    ix = healpy.ang2pix(nside, catalog.ra_deg, catalog.dec_deg, lonlat=True)
    catalog.add_column('stardens', m[ix])

    return catalog


def read_randoms(ix_list, download=False):
    """Returns randoms with no "quality cuts"."""

    assert len(ix_list) >= 1
    assert len(set(ix_list)) == len(ix_list)  # no duplicates
    
    catalog_list = [ ]
    
    for ix in ix_list:
        assert 0 <= ix < 200   # generalize later

        # List of pairs (my_col_name, fits_col_name).
        cols = [ ('ra_deg','RA'), ('dec_deg','DEC'), ('ebv','EBV') ]
        cols += [ (f'nobs_{x}', f'NOBS_{x.upper()}') for x in ['g','r','z'] ]
        cols += [ ('maskbits','MASKBITS') ]
    
        filename = _random_filename(ix, mflag=False, download=download)
        catalog = Catalog.from_fits(filename, cols)

        # lrg_mask
        filename2 = _random_filename(ix, mflag=True, download=download)
        ctmp = Catalog.from_fits(filename2, [('lrg_mask','lrg_mask')])
        catalog.add_column('lrg_mask', ctmp.lrg_mask)
        
        # stardens column
        m = read_stardens_map(download=download)
        nside = healpy.npix2nside(len(m))
        ix = healpy.ang2pix(nside, catalog.ra_deg, catalog.dec_deg, lonlat=True)
        catalog.add_column('stardens', m[ix])

        catalog_list.append(catalog)

    return Catalog.concatenate(catalog_list, name='DESILS-LRG randoms')


def apply_quality_cuts(catalog, min_nobs=2, max_ebv=0.15, max_stardens=2500, lrg_mask=True, island_mask=True):
    """Applies LRG quality cuts in-place to 'catalog' (usually called just after read_galaxies() or read_randoms()).

    References:
       https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/quality_cuts.py
       https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/randoms_quality_cuts.py
       https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/lrgmask_v1.1/README.txt
    """

    if island_mask:
        mask = ~((catalog.dec_deg < -10.5) & (catalog.ra_deg > 120) & (catalog.ra_deg < 260))
        catalog.apply_boolean_mask(mask, name = 'DESILS-LRG quality cut: remove NGC islands')

    if (min_nobs is not None) and (min_nobs > 0):
        mask = (catalog.nobs_g >= min_nobs) & (catalog.nobs_r >= min_nobs) & (catalog.nobs_z >= min_nobs)
        catalog.apply_boolean_mask(mask, name = f'DESILS-LRG quality cut: nobs >= {min_nobs}')

    if lrg_mask:
        bits = (1<<1) | (1<<12) | (1<<13)
        mask = np.logical_and(catalog.lrg_mask == 0, (catalog.maskbits & bits) == 0)
        catalog.apply_boolean_mask(mask, name = 'DESILS-LRG quality cut: LRG maskbits == 0')

    if max_ebv is not None:
        mask = (catalog.ebv < max_ebv)
        catalog.apply_boolean_mask(mask, name = f'DESILS-LRG quality cut: EBV < {max_ebv}')

    if max_stardens is not None:
        mask = (catalog.stardens < max_stardens)
        catalog.apply_boolean_mask(mask, name = f'DESILS-LRG quality cut: stardens < {max_stardens}')
    

@functools.cache
def read_stardens_map(download=False):
    """Returns nside=64 ring-ordered healpix map.

    Note that read_galaxies() and read_randoms() call this function 'under the hood',
    and store the results in the 'stardens' Catalog column, so you probably won't need
    to call this function directly."""
    
    filename = _desils_lrg_path('misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits', download=True)
    
    print(f'Reading {filename}')
    f = fitsio.read(filename)

    nside = 64
    npix = 12 * nside**2
    
    assert np.all(f['HPXPIXEL'] == np.arange(npix))
    return np.asarray(f['STARDENS'], dtype=np.float32)  # convert big-endian float32 -> native


####################################################################################################


def _catalog_filename(extended, suffix=None, download=False):
    """The 'extended' argument should be True or False.
    
    Catalogs are split across multiple FITS files, distinguished by 'suffix':

        suffix=None      TARGETID, RA, DEC, EBV, PIXEL_NOBS_{G,R,Z}, MASKBITS, 
                         PHOTSYS, Z_PHOT_MEDIAN, lrg_mask, pz_bin

        suffix='more_1'  GAIA_PHOT_{BP,RP}_MEAN_MAG, GAIA_ASTROMETRIC_EXCESS_NOISE,
                         FITBITS, FRACFLUX_{G,R,Z,W1,W2}, FRACMASKED_{G,R,Z},
                         FRACIN_{G,R,Z}, FIBERTOTFLUX_G, SHAPE_{R,R_IVAR},
                         SHAPE_{E1,E2}, SERSIC, DCHISSQ

        suffix='more_2'  GALDEPTH_{G,R,Z}, PSFDEPTH_{G,R,Z,W1,W2}, PSFSIZE_{G,R,Z}

        suffix='photom'  TYPE, EBC, FLUX_{G,R,Z,W1,W2}, FLUX_IVAR_{G,R,Z,W1,W2},
                         MW_TRANSMISSION_{G,R,Z,W1,W2}, FIBERFLUX_{G,R,Z},
                         FIBERTOTFLUX_{R,Z}, GAIA_PHOT_G_MEAN_MAG

        suffix='pz'      Z_PHOT_{MEAN,MEDIAN,STD,L68,U68,L95,U95}, 
                         Z_SPEC, SURVEY, TRAINING

        suffix='pzbins-weights'          weight
        suffix='pzbins-weights_no_ebv'   weight

    For descriptions of some of these columns, see:

        https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/README.txt        
        https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    """

    lrg = 'extended_lrg' if extended else 'lrg'
    slist = [ None, 'more_1', 'more_2', 'photom', 'pz', 'pzbins-weights', 'pzbins-weights_no_ebv' ]
    
    if suffix not in slist:
        raise RuntimeError(f"kszx.desils_lrg._catalog_filename: suffix='{suffix}' invalid."
                           + f" Valid suffixes are: {slist}")

    if suffix is None:
        return _desils_lrg_path(f'catalogs/dr9_{lrg}_pzbins.fits', download=download)
    else:
        return _desils_lrg_path(f'catalogs/more/dr9_{lrg}_{suffix}.fits', download=download)


def _random_filename(ix, mflag, download=False):
    """The index 'ix' satisfies 0 <= ix < 200.

    The randoms are split across two FITS files, distinguished through the boolean 'mflag' arg:
      - mflag=False   ETS file containing RA, DEC, NOBS_{G,R,Z}, EBV  (+ more cols I don't use)
      - mflag=True    LRG mask file containing boolean 'lrg_mask' col
    """

    assert 0 <= ix < 200
    s = f'{(ix//20)+1}-{(ix%20)}'   # E.g. (ix=31) -> '2-11'

    if mflag:
        return _desils_lrg_path(f'catalogs/lrgmask_v1.1/randoms-{s}-lrgmask_v1.1.fits.gz', download=download)
    else:
        return _desils_ets_path(f'target/catalogs/dr9/0.49.0/randoms/resolve/randoms-{s}.fits', download=download)
    
    
def _desils_lrg_path(relpath, download=False):
    """Example: _desils_lrs_path('catalogs/dr9_lrg_pzbins.fits').
        -> local filename /data/desils/lrg_xcorr_2023/v1/catalogs/dr9_lrg_pzbins.fits
        -> url https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/dr9_lrg_pzbins.fits

    Intended to be called through wrapper, e.g. _catalog_filename()."""
    
    relpath = os.path.join('lrg_xcorr_2023/v1', relpath)
    desils_base_dir = io_utils.get_data_dir('desils')
    abspath = os.path.join(desils_base_dir, relpath)

    if download and not os.path.exists(abspath):
        url = f'https://data.desi.lbl.gov/public/papers/c3/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
    
    return abspath


def _desils_ets_path(relpath, download=False):
    """Example: _desils_ets_path('target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits')
        -> local filename /data/desils/ets/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits
        -> url https://data.desi.lbl.gov/public/ets/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits

    Intended to be called through wrapper, e.g. _random_filename()."""

    relpath = os.path.join('ets', relpath)
    desils_base_dir = io_utils.get_data_dir('desils')
    abspath = os.path.join(desils_base_dir, relpath)

    if download and not os.path.exists(abspath):
        url = f'https://data.desi.lbl.gov/public/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
    
    return abspath
