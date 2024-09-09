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

# %%
import kszx
import local_pipeline
import numpy as np
import matplotlib.pyplot as plt

# %%
fnl = 250
bg = local_pipeline.bg
deltac = local_pipeline.deltac
nkbins = local_pipeline.nkbins
kbin_delim = local_pipeline.kbin_delim
kmid = (kbin_delim[1:] + kbin_delim[:-1]) / 2.
# print(kmid)

# %%
cosmo = kszx.io_utils.read_pickle('data/cosmology.pkl')
box = kszx.io_utils.read_pickle('data/bounding_box.pkl')
pk_mcs = kszx.io_utils.read_npy('data/pk_mcs.npy')  # (nmc, 4, 4, nkbins)
# print(f'{pk_mcs.shape=}')

# %% [markdown]
# ### MC power spectra (k-binned)

# %%
nmc = len(pk_mcs)

# shape (nmc, nkbins)
pgg_mcs = pk_mcs[:,0,0,:]
pgg_ng_mcs = pk_mcs[:,0,0,:] + 2*fnl*pk_mcs[:,0,1,:]
pgvfake_mcs = pk_mcs[:,0,2,:]
pgvfake_ng_mcs = pk_mcs[:,0,2,:] + fnl*pk_mcs[:,1,2,:]
pgv_mcs = -pk_mcs[:,0,3,:]
pgv_ng_mcs = -pk_mcs[:,0,3,:] - fnl*pk_mcs[:,1,3,:]

pgg = np.mean(pgg_mcs, axis=0)
pgg_ng = np.mean(pgg_ng_mcs, axis=0)
pgvfake = np.mean(pgvfake_mcs, axis=0)
pgvfake_ng = np.mean(pgvfake_ng_mcs, axis=0)
pgv = np.mean(pgv_mcs, axis=0)
pgv_ng = np.mean(pgv_ng_mcs, axis=0)

# Errorbars are so small that they're not visible in plots!
pgg_err = (np.var(pgg_mcs, axis=0))**0.5

# %% [markdown]
# ### Unwindowed power spectra (k-binned)

# %%
zstar = 0.57
faH_star = cosmo.frsd(z=zstar) * cosmo.H(z=zstar) / (1+zstar)

pth_gg = bg**2 * kszx.lss.kbin_average(box, lambda k: cosmo.Plin(k=k,z=zstar), kbin_delim)
pth_gg_x = 2 * deltac * bg * (bg-1) * kszx.lss.kbin_average(box, lambda k: cosmo.Plin(k=k,z=zstar)/cosmo.alpha(k=k,z=zstar), kbin_delim)
pth_gg_ng = pth_gg + 2*fnl * pth_gg_x

pth_gv = bg * faH_star * kszx.lss.kbin_average(box, lambda k: cosmo.Plin(k=k,z=zstar)/k, kbin_delim)
pth_gv_x = 2 * deltac * (bg-1) * faH_star * kszx.lss.kbin_average(box, lambda k: cosmo.Plin(k=k,z=zstar)/cosmo.alpha(k=k,z=zstar)/k, kbin_delim)
pth_gv_ng = pth_gv + fnl * pth_gv_x 

# %% [markdown]
# ### P_gg and its fNL derivative (arbitrary normalization, shot noise not included)

# %%
# FIXME: adjusted by eye to make curves agree!
# Figure out how to compute normalization properly.
ad_hoc_normalization = 4.8e-6

plt.plot(kmid, pgg, marker='o', markersize=5, label='fNL=0')
plt.plot(kmid, pgg_ng, marker='o', markersize=5, label='fNL=250')
plt.plot(kmid, ad_hoc_normalization * pth_gg, marker='x', markersize=7, ls='None', color='red', label='Unwindowed (fNL=0)')
plt.plot(kmid, ad_hoc_normalization * pth_gg_ng, marker='+', markersize=10, ls='None', color='red', label='Unwindowed (fNL=250)')
plt.xlabel(f'$k$ (Mpc)')
plt.ylabel('$P_{gg}(k)$ [arbitrary normalization]')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ### P_{g,vfake} and its fNL derivative (arbitrary normalization)

# %%
# FIXME: adjusted by eye to make curves agree!
# Figure out how to compute normalization properly.
ad_hoc_normalization = 1.0e-8

plt.plot(kmid, pgvfake, marker='o', markersize=5, label='fNL=0')
plt.plot(kmid, pgvfake_ng, marker='o', markersize=5, label='fNL=250')
plt.plot(kmid, ad_hoc_normalization * pth_gv, marker='x', markersize=7, ls='None', color='red', label='Unwindowed (fNL=0)')
plt.plot(kmid, ad_hoc_normalization * pth_gv_ng, marker='+', markersize=10, ls='None', color='red', label='Unwindowed (fNL=250)')
plt.xlabel(f'$k$ (Mpc)')
plt.ylabel('$P_{g,vfake}(k)$ [arbitrary normalization]')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ### P_{gv} and its fNL derivative (arbitrary normalization)

# %%
# FIXME: adjusted by eye to make curves agree!
# Figure out how to compute normalization properly.
ad_hoc_normalization = 2.0e-9

plt.plot(kmid, pgv, marker='o', markersize=5, label='fNL=0')
plt.plot(kmid, pgv_ng, marker='o', markersize=5, label='fNL=250')
plt.plot(kmid, ad_hoc_normalization * pth_gv, marker='x', markersize=7, ls='None', color='red', label='Unwindowed spin-0 (fNL=0)')
plt.plot(kmid, ad_hoc_normalization * pth_gv_ng, marker='+', markersize=10, ls='None', color='red', label='Unwindowed spin-0 (fNL=250)')
plt.xlabel(f'$k$ (Mpc)')
plt.ylabel('$P_{gv}(k)$ [arbitrary normalization]')
plt.legend(loc='upper right')
plt.show()

# %%
