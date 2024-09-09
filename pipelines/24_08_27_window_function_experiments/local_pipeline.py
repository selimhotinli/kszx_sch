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


####################################################################################################


def run_camb():
    cosmo = kszx.Cosmology('planck18+bao')
    kszx.io_utils.write_pickle('data/cosmology.pkl', cosmo)


def make_bounding_box():
    cosmo = kszx.io_utils.read_pickle('data/cosmology.pkl')
    rcat = kszx.sdss.read_randoms('CMASS_North')
    rcat.apply_redshift_cut(0.43, 0.7)

    bb = kszx.BoundingBox(rcat, cosmo, pixsize=pixsize, rpad=rpad)
    kszx.io_utils.write_pickle('data/bounding_box.pkl', bb)
    

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
    

def run_mc(output_filename):
    cosmo = kszx.io_utils.read_pickle('data/cosmology.pkl')
    box = kszx.io_utils.read_pickle('data/bounding_box.pkl')

    rcat = kszx.Catalog.from_h5('data/randoms_with_act_ivar.h5')
    rcat_D = cosmo.D(z=rcat.z, z0norm=True)
    rcat_f = cosmo.frsd(z=rcat.z)
    rcat_H = cosmo.H(z=rcat.z)
    rcat_r = cosmo.chi(z=rcat.z)
    rcat_xyz = kszx.utils.ra_dec_to_xyz(rcat.ra_deg, rcat.dec_deg, r=rcat_r)

    delta0 = kszx.lss.simulate_gaussian_field(box, cosmo.Plin_z0)

    # Catalog representation of Sg
    t = kszx.lss.fft_c2r(box, delta0)                                 # real-space delta0
    t = kszx.lss.interpolate_points(box, t, rcat_xyz, kernel=kernel)  # catalog delta0
    rcat_Sg = bg * rcat_D * t                                         # catalog delta_g

    # Catalog representation of dSg/dfNL
    fk = lambda k: 1.0 / cosmo.alpha(k=k,z=0)
    t = kszx.lss.multiply_kfunc(box, delta0, fk, dc=0)    # Fourier-space delta0/alpha0
    t = kszx.lss.fft_c2r(box, t)                          # real-space delta0/alpha0
    t = kszx.lss.interpolate_points(box, t, rcat_xyz, kernel=kernel)  # catalog delta0/alpha0
    rcat_dSg = 2 * deltac * (bg-1) * t                    # catalog d(deltag)/dfNL (note no factor D(z))

    # Catalog representation of vfake (scalar field with same power spectrum as v)
    t = kszx.lss.multiply_kfunc(box, delta0, lambda k:1/k, dc=0)     # Fourier-space delta0/k
    t = kszx.lss.fft_c2r(box, t)                                     # real-space delta0/k
    t = kszx.lss.interpolate_points(box, t, rcat_xyz, kernel=kernel)  # catalog delta0/k
    rcat_vfake = t * rcat_f * rcat_H * rcat_D / (1+rcat.z)            # catalog v = faHD/k delta0
    rcat_vfake *= rcat.act_ivar

    # Catalog representation of vr
    rcat_vr = np.zeros(rcat.size)
    u = kszx.lss.multiply_kfunc(box, delta0, lambda k:1/k**2, dc=0)  # Fourier-space delta0/k^2
    
    for axis in range(3):
        t = kszx.lss.apply_partial_derivative(box, u, axis=axis)
        t = kszx.lss.fft_c2r(box, t)
        t = kszx.lss.interpolate_points(box, t, rcat_xyz, kernel=kernel)  # catalog delta0/k
        rcat_vr += t * rcat_xyz[:,axis] / rcat_r
        
    rcat_vr *= rcat_f * rcat_H * rcat_D / (1+rcat.z)         # catalog v = faHD/k delta0
    rcat_vr *= rcat.act_ivar

    # vr -> v1
    map_v1 = np.zeros(box.fourier_space_shape, dtype=complex)

    for axis in range(3):
        t = np.zeros(box.real_space_shape, dtype=float)
        kszx.lss.grid_points(box, t, rcat_xyz, weights = rcat_vr * rcat_xyz[:,axis] / rcat_r, kernel=kernel)
        t = kszx.lss.fft_r2c(box, t)
        map_v1 += kszx.lss.apply_partial_derivative(box, t, axis, in_place=True)  # still need to divide by k

    # Divide by k
    map_v1 = kszx.lss.multiply_kfunc(box, map_v1, lambda k:1.0/k, dc=0., in_place=True)

    def rcat_to_map(w):
        t = np.zeros(box.real_space_shape, dtype=float)
        kszx.lss.grid_points(box, t, rcat_xyz, weights=w, kernel=kernel)
        return kszx.lss.fft_r2c(box, t)

    map_Sg = rcat_to_map(rcat_Sg)
    map_dSg = rcat_to_map(rcat_dSg)
    map_vfake = rcat_to_map(rcat_vfake)

    pk = kszx.lss.estimate_power_spectrum(box, [map_Sg,map_dSg,map_vfake,map_v1], kbin_delim, use_dc=False)
    kszx.io_utils.write_npy(output_filename, pk)


def aggregate_mcs():
    mcs = np.array([ kszx.io_utils.read_npy(f'data/mcs/pk_{imc}.npy') for imc in range(nmc) ])
    kszx.io_utils.write_npy(f'data/pk_mcs.npy', mcs)
