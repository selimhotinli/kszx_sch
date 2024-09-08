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
pk_mcs = kszx.io_utils.read_npy('data/pk_mcs.npy')
print(f'{pk_mcs.shape=}')

# %%
pk = np.mean(pk_mcs, axis=0)
pk_err = np.var(pk_mcs, axis=0)**0.5
print(pk.shape)
print(pk_err.shape)

# %%
nkbins = local_pipeline.nkbins
kbin_delim = local_pipeline.kbin_delim
kmid = (kbin_delim[1:] + kbin_delim[:-1]) / 2.
print(kmid)

# %% [markdown]
# ### P_gg and its fNL derivative (arbitrary normalization)

# %%
plt.plot(kmid, pk[0,0])
plt.plot(kmid, pk[0,0]+2*250*pk[0,1])

# %% [markdown]
# ### P_{g,vfake} and its fNL derivative (arbitrary normalization)

# %%
plt.plot(kmid, pk[0,2])
plt.plot(kmid, pk[0,2]+250*pk[1,2])

# %% [markdown]
# ### P_{gv} and its fNL derivative (arbitrary normalization)

# %%
plt.plot(kmid, -pk[0,3])
plt.plot(kmid, -pk[0,3]-250*pk[1,3])

# %%
