# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook creates catalogs `data/gcat.h5`, `rcat.h5` with the following columns:
#
# ```
# ra_deg, dec_deg            2-d coords
# z, zerr                    photo-z and error
# wmask, cmask               CMB masks (wide/cluster)
# t{90,150}_{cmask,nocmask}  filtered CMB temperature
# ```

# %%
import kszx
import healpy
import pixell
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Globals

# %%
extended = True  # use extended LRGs?
num_randcat_files = 5
act_dr = 5
filt_lmin = 2250
filt_lmax = 6000
filt_l0 = 4000

# Decided not to put redshift cut in .h5 catalogs
# zmin = 0.4
# zmax = 1.0

# %%
cosmo = kszx.Cosmology('planck18+bao')
kszx.io_utils.write_pickle('data/cosmology.pkl', cosmo)

# %% [markdown]
# # ACT inverse variance, masks

# %%
ivar90 = kszx.act.read_ivar(freq=90, dr=act_dr)
ivar150 = kszx.act.read_ivar(freq=150, dr=act_dr)

# %%
wide_mask = kszx.act.read_nilc_wide_mask()
cluster_mask = kszx.act.read_cluster_mask()

# %%
kszx.pixell_utils.plot_map(wide_mask, downgrade=30)
kszx.pixell_utils.plot_map(cluster_mask, downgrade=30)

# %%
kszx.pixell_utils.plot_map(ivar90, downgrade=30)
kszx.pixell_utils.plot_map(ivar150, downgrade=30)

# %% [markdown]
# # ACT CMB maps

# %%
cmb90 = kszx.act.read_cmb(freq=90, dr=act_dr)
cmb150 = kszx.act.read_cmb(freq=150, dr=act_dr)

# %%
kszx.pixell_utils.plot_map(wide_mask * cmb90, downgrade=30)
kszx.pixell_utils.plot_map(wide_mask * cmb150, downgrade=30)

# %% [markdown]
# # CMB filtering

# %%
ell = np.arange(filt_lmax+1, dtype=float)
Fl = np.where(ell >= filt_lmin, np.exp(-(ell/filt_l0)**2), 0)
plt.plot(ell, Fl)
plt.xlabel(r'$l$')
plt.ylabel(r'$F_l$')
plt.show()


# %%
def filter_cmb(cmb, ivar, cmask_flag):
    """pixell map -> pixell map"""
    
    m = wide_mask * ivar * cmb
    if cmask_flag:
        m *= cluster_mask
    
    alm = kszx.pixell_utils.map2alm(m, lmax=filt_lmax)
    alm = healpy.almxfl(alm, Fl)
    return kszx.pixell_utils.alm2map(alm, wide_mask.shape, wide_mask.wcs)


# %%
fcmb90_nocmask = filter_cmb(cmb90, ivar90, cmask_flag=False)
kszx.pixell_utils.plot_map(fcmb90_nocmask, downgrade=30)

fcmb90_cmask = filter_cmb(cmb90, ivar90, cmask_flag=True)
kszx.pixell_utils.plot_map(fcmb90_cmask, downgrade=30)

kszx.pixell_utils.plot_map(fcmb90_nocmask - fcmb90_cmask, downgrade=30)

# %%
fcmb150_nocmask = filter_cmb(cmb150, ivar150, cmask_flag=False)
kszx.pixell_utils.plot_map(fcmb150_nocmask, downgrade=30)

fcmb150_cmask = filter_cmb(cmb150, ivar150, cmask_flag=True)
kszx.pixell_utils.plot_map(fcmb150_cmask, downgrade=30)

kszx.pixell_utils.plot_map(fcmb150_nocmask - fcmb150_cmask, downgrade=30)

# %% [markdown]
# # DESILS catalogs

# %%
gcat = kszx.desils_lrg.read_galaxies(extended=extended)
kszx.desils_lrg.apply_quality_cuts(gcat)
# gcat.apply_redshift_cut(zmin, zmax)  # decided not to put redshift cut in .5 catalogs

# %%
rcat = kszx.desils_lrg.read_randoms(range(5))
kszx.desils_lrg.apply_quality_cuts(rcat)

# %%
gmap = kszx.pixell_utils.map_from_catalog(wide_mask.shape, wide_mask.wcs, gcat=gcat, rcat=rcat)

# %%
kszx.pixell_utils.plot_map(gmap, downgrade=30)

# %% [markdown]
# # Augment catalogs

# %%
# After applying quality cuts, these cols are no longer needed.
for col in [ 'ebv', 'lrg_mask', 'nobs_g', 'nobs_r', 'nobs_z', 'maskbits', 'stardens' ]:
    gcat.remove_column(col)
    rcat.remove_column(col)

print(f'{gcat.col_names = }')
print(f'{rcat.col_names = }')

# %%
# Add (z,zerr) to rcat.

ix = np.random.randint(low=0, high=gcat.size, size=rcat.size)
rcat.add_column('z', gcat.z[ix])
rcat.add_column('zerr', gcat.zerr[ix])

print(f'{gcat.col_names = }')
print(f'{rcat.col_names = }')

# %%
# These histograms should look the same

plt.hist(gcat.z, bins=100)
plt.show()

plt.hist(rcat.z, bins=100)
plt.show()


# %%
def _augment_catalog(catalog, col_name, pixell_map):
    print(col_name)
    m, _mask = kszx.pixell_utils.eval_map_on_catalog(pixell_map, catalog)
    catalog.add_column(col_name, m)

def augment_catalog(catalog):
    _augment_catalog(catalog, 'wmask', wide_mask)
    _augment_catalog(catalog, 'cmask', cluster_mask)
    _augment_catalog(catalog, 't90_nocmask', fcmb90_nocmask)
    _augment_catalog(catalog, 't90_cmask', fcmb90_cmask)
    _augment_catalog(catalog, 't150_nocmask', fcmb150_nocmask)
    _augment_catalog(catalog, 't150_cmask', fcmb150_cmask)


# %%
augment_catalog(gcat)
gcat.write_h5('data/gcat.h5')

# %%
augment_catalog(rcat)
rcat.write_h5('data/rcat.h5')

# %%
