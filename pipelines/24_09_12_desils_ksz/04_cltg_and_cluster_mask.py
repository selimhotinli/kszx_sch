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
# # Setup

# %%
import kszx
import healpy
import pixell
import numpy as np
import matplotlib.pyplot as plt

# %%
zmin = 0.4
zmax = 0.8
dr = 5

nlbins = 30
lmin = 2000
lmax = 6000
ldelim = np.linspace(lmin, lmax, nlbins+1)
lmid = (ldelim[:-1] + ldelim[1:]) / 2.

# %% [markdown]
# # Input files

# %%
gcat = kszx.Catalog.from_h5('data/gcat.h5')
rcat = kszx.Catalog.from_h5('data/rcat.h5')

# %%
wmask = kszx.act.read_nilc_wide_mask()

# %%
cmask = kszx.act.read_cluster_mask()

# %%
cmb90 = kszx.act.read_cmb(freq=90, dr=dr)
cmb150 = kszx.act.read_cmb(freq=150, dr=dr)

# %%
ivar90 = kszx.act.read_ivar(freq=90, dr=dr)
ivar150 = kszx.act.read_ivar(freq=150, dr=dr)

# %%
kszx.pixell_utils.plot_map(wmask, downgrade=30)

# %%
kszx.pixell_utils.plot_map(cmask, downgrade=30)

# %% [markdown]
# # Galaxy - random map

# %%
gcat.apply_redshift_cut(zmin,zmax)
rcat.apply_redshift_cut(zmin,zmax)

# %%
gmap = kszx.pixell_utils.map_from_catalog(wmask.shape, wmask.wcs, gcat=gcat, rcat=rcat)

# %%
kszx.pixell_utils.plot_map(gmap, downgrade=30)

# %%
# catalogs no longer needed!
del gcat, rcat

# %% [markdown]
# # Harmonic-space maps (alm)

# %%
glm = kszx.pixell_utils.map2alm(gmap, lmax=lmax)


# %%
def plot_and_transform(m):
    kszx.pixell_utils.plot_map(m, downgrade=30)
    return kszx.pixell_utils.map2alm(m, lmax=lmax)


# %%
tlm_90_nocmask = plot_and_transform(wmask * ivar90 * cmb90)
tlm_90_cmask = plot_and_transform(wmask * cmask * ivar90 * cmb90)
tlm_150_nocmask = plot_and_transform(wmask * ivar150 * cmb150)
tlm_150_cmask = plot_and_transform(wmask * cmask * ivar150 * cmb150)

# %% [markdown]
# # CMB-galaxy power spectra (ClTg)

# %%
cltg_90_nocmask = kszx.cmb.estimate_cl([glm,tlm_90_nocmask], lbin_delim=ldelim)[0,1,:]
cltg_90_cmask = kszx.cmb.estimate_cl([glm,tlm_90_cmask], lbin_delim=ldelim)[0,1,:]
cltg_150_nocmask = kszx.cmb.estimate_cl([glm,tlm_150_nocmask], lbin_delim=ldelim)[0,1,:]
cltg_150_cmask = kszx.cmb.estimate_cl([glm,tlm_150_cmask], lbin_delim=ldelim)[0,1,:]

# %%
plt.plot(lmid, lmid**2 * cltg_90_cmask, label='90 GHz, cluster mask')
plt.plot(lmid, lmid**2 * cltg_90_nocmask, label='90 GHz, no cluster mask')
plt.axhline(0, color='r', ls=':')
plt.legend(loc='lower right')
plt.xlabel('$l$')
plt.ylabel('$l^2 C_l^{Tg}$')
plt.title(r'CMB-galaxy cross power $l^2 C_l^{Tg}$, 90 GHz') 
plt.show()

# %%
plt.plot(lmid, lmid**2 * cltg_150_cmask, label='150 GHz, cluster mask')
plt.plot(lmid, lmid**2 * cltg_150_nocmask, label='150 GHz, no cluster mask')
plt.axhline(0, color='r', ls=':')
plt.legend(loc='lower right')
plt.xlabel('$l$')
plt.ylabel('$l^2 C_l^{Tg}$')
plt.title(r'CMB-galaxy cross power $l^2 C_l^{Tg}$, 150 GHz') 
plt.show()

# %% [markdown]
# # CMB-mask power spectra

# %%
kszx.pixell_utils.plot_map((1-cmask), downgrade=40)
kszx.pixell_utils.plot_map((1-cmask) * wmask, downgrade=40, colorbar=True)
plt.show()

# %%
maskalm = kszx.pixell_utils.map2alm((1-cmask) * wmask, lmax=lmax)

# %%
cl_mask_t90 = kszx.cmb.estimate_cl([maskalm,tlm_90_nocmask], lbin_delim=ldelim)[0,1,:]
cl_mask_t150 = kszx.cmb.estimate_cl([maskalm,tlm_150_nocmask], lbin_delim=ldelim)[0,1,:]

# %%
plt.plot(lmid, lmid**2 * cl_mask_t90, label='90 GHz')
plt.plot(lmid, lmid**2 * cl_mask_t150, label='150 GHz')
plt.axhline(0, color='r', ls=':')
plt.legend(loc='upper right')
plt.xlabel(r'$l$')
plt.ylabel(r'$l^2 C_l^{T,mask}$')
plt.title('CMB-mask correlation (expect negative sign for tSZ)')
plt.show()

# %%
cl_mask_g = kszx.cmb.estimate_cl([maskalm,glm], lbin_delim=ldelim)[0,1,:]

# %%
plt.plot(lmid, lmid**2 * cl_mask_g)
plt.axhline(0, color='r', ls=':')
plt.xlabel(r'$l$')
plt.ylabel(r'$l^2 C_l^{gal,mask}$')
plt.title('Galaxy-mask correlation (expect positive sign)')
plt.show()

# %%
