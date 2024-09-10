import kszx
import numpy as np


bg = 2
deltac = 1.42
kmax = 0.1
nkbins = 20
kbin_delim = np.linspace(0,kmax,nkbins+1)
nmc = 1000
kernel = 'cubic'
pixsize = 10.0
rpad = 200.0
sdss_survey = 'CMASS_North'
sdss_zmin = 0.43
sdss_zmax = 0.7
act_freq = 150
act_dr = 5


####################################################################################################


def run_camb():
    cosmo = kszx.Cosmology('planck18+bao')
    kszx.io_utils.write_pickle('data/cosmology.pkl', cosmo)


def make_bounding_box():
    cosmo = kszx.io_utils.read_pickle('data/cosmology.pkl')
    rcat = kszx.sdss.read_randoms('CMASS_North')
    rcat.apply_redshift_cut(sdss_zmin, sdss_zmax)

    bb = kszx.BoundingBox(rcat.get_xyz(cosmo), pixsize=pixsize, rpad=rpad)
    kszx.io_utils.write_pickle('data/bounding_box.pkl', bb)
    

def eval_act_ivar_on_sdss_randoms():
    rcat = kszx.sdss.read_randoms('CMASS_North')
    rcat.apply_redshift_cut(sdss_zmin, sdss_zmax)
    
    ivar = kszx.act.read_ivar(freq=act_freq, dr=act_dr)
    idec, ira, mask = kszx.pixell_utils.ang2pix(ivar.shape, ivar.wcs, rcat.ra_deg, rcat.dec_deg, allow_outliers=True)
    rcat_ivar = np.where(mask, ivar[idec,ira], 0)
    rcat.add_column('act_ivar', rcat_ivar)

    # Remove randoms from catalog if ACT does not observe the corresponding pixel.
    rcat = rcat.apply_boolean_mask(rcat_ivar > 0, name = 'ACT footprint')
    rcat.write_h5('data/randoms_with_act_ivar.h5')
    

def run_mc(output_filename):
    cosmo = kszx.io_utils.read_pickle('data/cosmology.pkl')
    box = kszx.io_utils.read_pickle('data/bounding_box.pkl')

    rcat = kszx.Catalog.from_h5('data/randoms_with_act_ivar.h5')
    rcat_D = cosmo.D(z=rcat.z, z0norm=True)
    rcat_f = cosmo.frsd(z=rcat.z)
    rcat_H = cosmo.H(z=rcat.z)
    rcat_xyz = rcat.get_xyz(cosmo)

    delta0 = kszx.lss.simulate_gaussian_field(box, cosmo.Plin_z0)

    # Catalog representation of Sg
    rcat_Sg = kszx.lss.interpolate_points(box, delta0, rcat_xyz, kernel=kernel, fft=True)
    rcat_Sg *= bg * rcat_D 

    # Catalog representation of dSg/dfNL
    fk = lambda k: 1.0 / cosmo.alpha(k=k,z=0)
    t = kszx.lss.multiply_kfunc(box, delta0, fk, dc=0)    # Fourier-space delta0/alpha0
    rcat_dSg = kszx.lss.interpolate_points(box, t, rcat_xyz, kernel=kernel, fft=True)
    rcat_dSg *= 2 * deltac * (bg-1)                       # note no factor D(z)

    # Catalog representation of vfake (scalar field with same power spectrum as v)
    t = kszx.lss.multiply_kfunc(box, delta0, lambda k:1/k, dc=0)     # Fourier-space delta0/k
    rcat_vfake = kszx.lss.interpolate_points(box, t, rcat_xyz, kernel=kernel, fft=True)
    rcat_vfake *= rcat_f * rcat_H * rcat_D / (1+rcat.z)             # catalog v = faHD/k delta0
    rcat_vfake *= rcat.act_ivar

    # Catalog representation of vr
    rcat_vr = kszx.lss.interpolate_points(box, t, rcat_xyz, kernel=kernel, fft=True, spin=1)
    rcat_vr *= rcat_f * rcat_H * rcat_D / (1+rcat.z)         # catalog v = faHD/k delta0
    rcat_vr *= rcat.act_ivar

    map_v1 = kszx.lss.grid_points(box, rcat_xyz, weights=rcat_vr, kernel=kernel, fft=True, spin=1)
    map_Sg = kszx.lss.grid_points(box, rcat_xyz, weights=rcat_Sg, kernel=kernel, fft=True)
    map_dSg = kszx.lss.grid_points(box, rcat_xyz, weights=rcat_dSg, kernel=kernel, fft=True)
    map_vfake = kszx.lss.grid_points(box, rcat_xyz, weights=rcat_vfake, kernel=kernel, fft=True)

    pk = kszx.lss.estimate_power_spectrum(box, [map_Sg,map_dSg,map_vfake,map_v1], kbin_delim, use_dc=False)
    kszx.io_utils.write_npy(output_filename, pk)


def aggregate_mcs():
    mcs = np.array([ kszx.io_utils.read_npy(f'data/mcs/pk_{imc}.npy') for imc in range(nmc) ])
    kszx.io_utils.write_npy(f'data/pk_mcs.npy', mcs)
