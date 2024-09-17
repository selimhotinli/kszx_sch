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
import numpy as np
import matplotlib.pyplot as plt

# %%
nside = 32

# %%
cosmo = kszx.io_utils.read_pickle('data/cosmology.pkl')
gcat = kszx.Catalog.from_h5('data/gcat.h5')
rcat = kszx.Catalog.from_h5('data/rcat.h5')


# %%
def analyze_zslice(zmin, zmax):
    gcat_tmp = gcat.apply_redshift_cut(zmin, zmax, in_place=False)
    rcat_tmp = rcat.apply_redshift_cut(zmin, zmax, in_place=False)
    m_tmp = kszx.healpix_utils.map_from_catalog(nside, gcat=gcat_tmp, rcat=rcat_tmp)
    healpy.mollview(m_tmp)
    plt.show()
    del gcat_tmp, rcat_tmp


# %% [markdown]
# # 0.4 < z < 0.5

# %%
analyze_zslice(0.4, 0.5)

# %% [markdown]
# # 0.5 < z < 0.6

# %%
analyze_zslice(0.5, 0.6)

# %% [markdown]
# # 0.6 < z < 0.7

# %%
analyze_zslice(0.6, 0.7)

# %% [markdown]
# # 0.7 < z < 0.8

# %%
analyze_zslice(0.7, 0.8)

# %% [markdown]
# # 0.8 < z < 0.9

# %%
analyze_zslice(0.8, 0.9)

# %% [markdown]
# # 0.9 < z < 1.0

# %%
analyze_zslice(0.9, 1.0)

# %% [markdown]
# # 1.0 < z < 1.1

# %%
analyze_zslice(1.0, 1.1)

# %%
ivar = kszx.act.read_ivar(freq=150,dr=5)

# %%
