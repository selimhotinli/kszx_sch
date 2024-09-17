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

pixsize = 10
rpad = 200
kernel = 'cubic'

nkbins = 20
kmax = 0.1
kbins = np.linspace(0, kmax, nkbins+1)
kmid = (kbins[1:] + kbins[:-1]) / 2.

# %% [markdown]
# # Input files

# %%
cosmo = kszx.io_utils.read_pickle('data/cosmology.pkl')
gcat = kszx.Catalog.from_h5('data/gcat.h5')
rcat = kszx.Catalog.from_h5('data/rcat.h5')

print(f'\n{gcat.size = }')
print(f'{rcat.size = }')

# %%
gcat.apply_redshift_cut(zmin, zmax)
rcat.apply_redshift_cut(zmin, zmax)

print(f'\n{gcat.size = }')
print(f'{rcat.size = }')

# %%
m = kszx.healpix_utils.map_from_catalog(nside=32, gcat=gcat, rcat=rcat)
healpy.mollview(m, title='galaxies minus randoms')
plt.show()

# %% [markdown]
# # 3d coords, Bounding box

# %%
gcat_xyz = gcat.get_xyz(cosmo)
rcat_xyz = rcat.get_xyz(cosmo)
gr_xyz = np.concatenate((gcat_xyz, rcat_xyz))

# %%
box = kszx.BoundingBox(gr_xyz, pixsize = pixsize, rpad = rpad)
print(box)

# %% [markdown]
# # Grid (+FFT) galaxies

# %% [markdown]
# Reminder: gcat/rcat have the following columns:
#
# ```
# ra_deg, dec_deg            2-d coords
# z, zerr                    photo-z and error
# wmask, cmask               CMB masks (wide/cluster)
# t{90,150}_{cmask,nocmask}  filtered CMB temperature
# ```

# %%
gmap_nowmask = kszx.lss.grid_points(
    box, gcat_xyz, rpoints=rcat_xyz, 
    kernel=kernel, fft=True)

gmap_wmask = kszx.lss.grid_points(
    box, gcat_xyz, weights=gcat.wmask, 
    rpoints=rcat_xyz, rweights=rcat.wmask,
    kernel=kernel, fft=True)

# %%
pgg_mat = kszx.lss.estimate_power_spectrum(box, [gmap_nowmask,gmap_wmask], kbins)
pgg_nowmask = pgg_mat[0,0,:]
pgg_wmask = pgg_mat[1,1,:]

# %%
r = np.sum(gcat.wmask) / gcat.size
plt.semilogy(kmid, pgg_nowmask)
plt.semilogy(kmid, pgg_wmask/r)
plt.show()


# %% [markdown]
# # KSZ velocity reconstruction

# %% [markdown]
# Reminder: gcat/rcat have the following columns:
#
# ```
# ra_deg, dec_deg            2-d coords
# z, zerr                    photo-z and error
# wmask, cmask               CMB masks (wide/cluster)
# t{90,150}_{cmask,nocmask}  filtered CMB temperature
# ```

# %%
def make_vrec(gweights, rweights=None):
    assert gweights.shape == (gcat.size,)
    assert (rweights is None) or (rweights.shape == (rcat.size,))

    if rweights is None:
        xyz = gcat_xyz
        w = gweights
    else:
        xyz = gr_xyz
        w = np.concatenate((gweights,rweights))
        w[gcat.size:] *= -(gcat.size/rcat.size)

    # Note spin=1
    return kszx.lss.grid_points(box, xyz, weights=w, kernel=kernel, fft=True, spin=1)


# %% [markdown]
# # Are rweights useful?

# %%
vrec_norweights = make_vrec(gcat.t150_cmask)
vrec_rweights = make_vrec(gcat.t150_cmask, rcat.t150_cmask)

pvv_norweights = kszx.lss.estimate_power_spectrum(box, vrec_norweights, kbins)
pvv_rweights = kszx.lss.estimate_power_spectrum(box, vrec_rweights, kbins)

plt.semilogy(kmid, pvv_norweights, label='no r-weights')
plt.semilogy(kmid, pvv_rweights, label='with r-weights')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# # P_vv with and without cluster mask

# %%
vrec_90_nocmask = make_vrec(gcat.t90_nocmask)
vrec_90_cmask = make_vrec(gcat.t90_cmask)
vrec_150_nocmask = make_vrec(gcat.t150_nocmask)
vrec_150_cmask = make_vrec(gcat.t150_cmask)

# %%
pvv_90_nocmask = kszx.lss.estimate_power_spectrum(box, vrec_90_nocmask, kbins)
pvv_90_cmask = kszx.lss.estimate_power_spectrum(box, vrec_90_cmask, kbins)

plt.semilogy(kmid, pvv_90_cmask, label='90 GHz, with cluster mask')
plt.semilogy(kmid, pvv_90_nocmask, label='90 GHz, no cluster mask')
plt.legend(loc='upper right')
plt.xlabel(r'$k$ (Mpc$^{-1}$)')
plt.ylabel(r'$P_{vv}(k)$')
plt.title(r'Total kSZ reconstruction power $P{vv}(k)$, arbitrary normalization')
plt.show()

# %% jupyter={"source_hidden": true}
pvv_150_nocmask = kszx.lss.estimate_power_spectrum(box, vrec_150_nocmask, kbins)
pvv_150_cmask = kszx.lss.estimate_power_spectrum(box, vrec_150_cmask, kbins)

plt.semilogy(kmid, pvv_150_cmask, label='150 GHz, with cluster mask')
plt.semilogy(kmid, pvv_150_nocmask, label='150 GHz, no cluster mask')
plt.legend(loc='upper right')
plt.xlabel(r'$k$ (Mpc$^{-1}$)')
plt.ylabel(r'$P_{vv}(k)$')
plt.title(r'Total kSZ reconstruction power $P{vv}(k)$, arbitrary normalization')
plt.show()

# %% [markdown]
# # P_gv (no cluster mask for now)

# %%
maps = [ gmap_nowmask, gmap_wmask, vrec_90_nocmask, vrec_150_nocmask ]
pmat = kszx.lss.estimate_power_spectrum(box, maps, kbins)

pgv_90_nowmask = pmat[0,2,:]
pgv_90_wmask = pmat[1,2,:]
pgv_150_nowmask = pmat[0,3,:]
pgv_150_wmask = pmat[1,3,:]

# %%
plt.plot(kmid, kmid**2 * pgv_90_nowmask, 
         label='90 GHz, no CMB mask applied to galaxy survey')

plt.plot(kmid, kmid**2 * pgv_90_wmask,
        label='90 GHz, CMB mask applied to galaxy survey')

plt.axhline(0, color='r', ls=':')
plt.legend(loc='lower right')

plt.xlabel(r'$k$ (Mpc$^{-1}$)')
plt.ylabel(r'$k^2 P_{gv}(k)$')
plt.title(r'$k^2 P{gv}(k)$, 90 GHz, arbitrary normalization')
plt.show()

# %%
plt.plot(kmid, kmid**2 * pgv_150_nowmask,
          label='150 GHz, no CMB mask applied to galaxy survey')

plt.plot(kmid, kmid**2 * pgv_150_wmask,
        label='150 GHz, CMB mask applied to galaxy survey')

plt.axhline(0, color='r', ls=':')
plt.legend(loc='lower right')

plt.xlabel(r'$k$ (Mpc$^{-1}$)')
plt.ylabel(r'$k^2 P_{gv}(k)$')
plt.title(r'$k^2 P{gv}(k)$, 150 GHz, arbitrary normalization')
plt.show()


# %%
