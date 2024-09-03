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
import numpy as np
import matplotlib.pyplot as plt
import kszx

# %% [markdown]
# ### Read ACT ivar map (150 GHz) and SDSS randoms (CMASS_North)

# %%
ivar = kszx.act.read_ivar(freq=150, dr=5)
rcat = kszx.Catalog.from_h5('data/randoms_with_act_ivar.h5')

# %% [markdown]
# ### Plot ACT ivar map
# This includes sky regions not covered by SDSS.

# %%
kszx.pixell_utils.plot_map(ivar, downgrade=40)

# %% [markdown]
# ### Plot SDSS randoms with ACT ivar weighting
# This defines the survey geometry for windowed power spectra, in later pipeline stages.

# %%
rmap = rcat.to_pixell(ivar.shape, ivar.wcs, weights=rcat.act_ivar)
kszx.pixell_utils.plot_map(rmap, downgrade=40)

# %% [markdown]
# ### Redshift distribution of randoms

# %%
plt.hist(rcat.z, bins=50)
plt.xlabel(r'Redshift $z$')
plt.show()

print(f'Min redshift: {np.min(rcat.z)}')
print(f'Max redshift: {np.max(rcat.z)}')

# %% [markdown]
# ### Bounding box

# %%
bb = kszx.io_utils.read_pickle('data/bounding_box.pkl')
print(bb)

# %%
