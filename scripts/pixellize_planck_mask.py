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
import healpy
import numpy as np
import matplotlib.pyplot as plt
import pixell.reproject

# %% [markdown]
# ## Unmasked CMB

# %%
cmb = kszx.act.read_cmb(freq=150, dr=5)

# %%
kszx.pixell_utils.plot_map(cmb, downgrade=40)


# %% [markdown]
# ## fsky=0.8

# %%
def pixellize_one_mask(sky_percentage, apodization=0):
    output_filename = f'pixellized_galmask_{sky_percentage}_apo{apodization}.fits'
    healpix_mask = kszx.planck.read_hfi_galmask(sky_percentage=sky_percentage, apodization=apodization)
    healpy.mollview(healpix_mask)
    plt.show()

    # NOTE rot='gal,equ'
    # NOTE method='spline' and order=0
    print('Pixellizing mask -- this will take some time')
    pixell_mask = pixell.reproject.healpix2map(healpix_mask, cmb.shape, cmb.wcs, rot='gal,equ', method="spline", order=0)
    
    kszx.pixell_utils.plot_map(pixell_mask, downgrade=40)
    kszx.pixell_utils.plot_map(cmb*pixell_mask, downgrade=40)
    kszx.pixell_utils.write_map(output_filename, pixell_mask)


# %%
pixellize_one_mask(80)

# %% [markdown]
# ## fsky=0.7

# %%
pixellize_one_mask(70)
