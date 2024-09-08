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
pk_mcs = kszx.io_utils.read_npy('data/pk_mcs.npy')  # (nmc, 4, 4, nkbins)
# print(f'{pk_mcs.shape=}')

# %%
nmc = len(pk_mcs)

# shape (nmc, nkbins)
pgg_mcs = pk_mcs[:,0,0,:]
pgg_ng_mcs = pk_mcs[:,0,0,:] + 2*fnl*pk_mcs[:,0,1,:]
pgvf_mcs = pk_mcs[:,0,2,:]
pgvf_ng_mcs = pk_mcs[:,0,2,:] + fnl*pk_mcs[:,1,2,:]
pgv_mcs = -pk_mcs[:,0,3,:]
pgv_ng_mcs = -pk_mcs[:,0,3,:] - fnl*pk_mcs[:,1,3,:]

pgg = np.mean(pgg_mcs, axis=0)
pgg_ng = np.mean(pgg_ng_mcs, axis=0)
pgvf = np.mean(pgvf_mcs, axis=0)
pgvf_ng = np.mean(pgvf_ng_mcs, axis=0)
pgv = np.mean(pgv_mcs, axis=0)
pgv_ng = np.mean(pgv_ng_mcs, axis=0)

# Errorbars are so small that they're not visible in plots!
pgg_err = (np.var(pgg_mcs, axis=0))**0.5

# %% [markdown]
# ### P_gg and its fNL derivative (arbitrary normalization)

# %%
plt.plot(kmid, pgg, marker='o', markersize=5, label='fNL=0')
plt.plot(kmid, pgg_ng, marker='o', markersize=5, label='fNL=250')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ### P_{g,vfake} and its fNL derivative (arbitrary normalization)

# %%
plt.plot(kmid, pgvf, marker='o', markersize=5, label='fNL=0')
plt.plot(kmid, pgvf_ng, marker='o', markersize=5, label='fNL=250')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ### P_{gv} and its fNL derivative (arbitrary normalization)

# %%
plt.plot(kmid, pgv, marker='o', markersize=5, label='fNL=0')
plt.plot(kmid, pgv_ng, marker='o', markersize=5, label='fNL=250')
plt.legend(loc='upper right')
plt.show()

# %%
