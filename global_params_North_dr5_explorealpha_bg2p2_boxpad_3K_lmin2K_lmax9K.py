# "Global" parameters, used in multiple notebooks,
# This .py file is imported into .ipynb notebooks with 'import global_params'

import numpy as np

# This is the output dir for the "initial pipeline",
# and the input dir for the "P(k) pipeline".
pipeline_indir  = 'pipelines/ks_08_06_inputs_North_dr5_explorealpha_bg2p2_boxpad_3K_lmin2K_lmax9K'
pipeline_outdir = 'pipelines/ks_08_06_outputs_North_dr5_explorealpha_bg2p2_boxpad_3K_lmin2K_lmax9K'

# Used in a lot of places
ksz_lmin = 2000
ksz_lmax = 9000
nkbins = 25
kmax = 0.05
kbin_edges = np.linspace(0, kmax, nkbins+1)
kbin_centers = (kbin_edges[1:] + kbin_edges[:-1]) / 2.

# KSZ
apply_cmb_mask_to_galaxy_catalog = True
surr_bg = 2.2   # galaxy bias used in surrogate sims
nsurr = 1500     # number of surrogate sims

# DESILS-LRG params
desils_lrg_zmin = 0.4
desils_lrg_zeff = 0.7
desils_lrg_zmax = 1.0
desils_lrg_dec_min = -55.0  # degrees
desils_lrg_dec_max = 20.0   # degrees
desils_lrg_north_only = True    # restrict to Galactic north? (for now, defined by 105 < ra_deg < 280)
desils_lrg_south_only = False   # restrict to Galactic south?
desils_lrg_extended = True
desils_lrg_randcat_files = 5
desils_lrg_rpad = 3000   # Mpc (used for bounding box)
desils_lrg_pixsize = 15  # Mpc (used for bounding box)

# Halo model params, to compute P_ge^{fid} and P_gg^{fid}.
# From Selim notebook
hmodel_ks = np.geomspace(1e-5,100,1000)
hmodel_ms = np.geomspace(2e10,1e17,40)
hmodel_minz = 0.4   # from Selim notebook
hmodel_maxz = 1.0   # from Selim notebook
hmodel_zeff = 0.7   # from Selim notebook
hmodel_ngal = 0.00033582 # rough CMASS number density Mpc^-3

# Planck galmask
galmask_sky_percentage = 70

# ACT data.
# The 'act_rms_threshold' param is used to define a boolean mask,
# by masking all pixels above the noise threshold.

act_dr = 5
act_rms_threshold = { 90: 70.0, 150: 70.0 }   # freq -> uK_arcmin
act_equalize_pixel_weighting = True       # use same Wcmb(theta) for 90 and 150?
act_equalize_l_weighting = True           # use F_l^{150} = (b_l^{90} / b_l^{150}) * F_l^{90}?
act_apply_cluster_mask = False            # not recommended (gives weird results, not clear why0
act_cluster_mask_apodization = 20         # pixels, not arcmin! Only used if apply_cluster_mask=True

# Deconvolution parameters. (See kszx.ksz_desils.RegulatedDeconvolver docstring.)
deconv_zbin_width = 0.01
deconv_soft_zmax = 1.5

# We downweight galaxies with large photo-z errors, using weighting
#  W = exp(-(zerr/(1+z))**2 / alpha)

alpha_g = 2.5e-3  # zerr-weight applied to delta_g (can be None)
alpha_vr = 2.5e-3 # zerr-weight applied to vrec (can be None)

# Number of redshift bins used for field-level mean-subtraction.
# Passed as 'nbins' argument to kszx.utils.subtract_binned_means().
# If zero, then kszx.utils.subtract_binned_means() is not called.

nzbins_gal = 25   # mean-subtraction for galaxy field
nzbins_vr = 25    # mean-subtraction for kSZ vrec reconstruction.
