import kszx
import numpy as np

def run_camb():
    cosmo = kszx.Cosmology('planck18+bao')
    kszx.io_utils.write_pickle('data/cosmology.pkl', cosmo)


def eval_act_ivar_on_sdss_randoms():
    rcat = kszx.sdss.read_randoms('CMASS_North')
    rcat.apply_redshift_cut(0.43, 0.7)
    
    ivar = kszx.act.read_ivar(freq=150, dr=5)
    idec, ira, mask = kszx.pixell_utils.ang2pix(ivar.shape, ivar.wcs, rcat.ra_deg, rcat.dec_deg, allow_outliers=True)
    rcat_ivar = np.where(mask, ivar[idec,ira], 0)
    rcat.add_column('act_ivar', rcat_ivar)

    # Remove randoms from catalog if ACT does not observe the corresponding pixel.
    rcat = rcat.apply_boolean_mask(rcat_ivar > 0, name = 'ACT footprint')
    rcat.write_h5('data/randoms_with_act_ivar.h5')
    
