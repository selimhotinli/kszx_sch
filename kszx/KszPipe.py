import os
import time
import yaml
import shutil
import functools
import numpy as np
import scipy.linalg
import scipy.special
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from . import core
from . import utils
from . import io_utils
from . import wfunc_utils

from .Box import Box
from .Catalog import Catalog
from .Cosmology import Cosmology
from .CatalogGridder import CatalogGridder
from .SurrogateFactory import SurrogateFactory

    
class KszPSE2:
    def __init__(self, box, cosmo, randcat, kbin_edges, surr_ngal_mean, surr_ngal_rms, surr_bg, rweights=None, nksz=0, ksz_rweights=None, ksz_bv=None, ksz_tcmb_realization=None, ztrue_col='z', zobs_col='z', deltac=1.68, kernel='cubic', surr_ic_nbins=1, use_dc=False, spin0_hack=False):
        assert isinstance(box, Box)
        assert isinstance(cosmo, Cosmology)
        assert isinstance(randcat, Catalog)
        assert kernel in [ 'cic', 'cubic' ]  # more kernels will be implemented soon!

        assert randcat.size >= 1000
        assert 'ra_deg' in randcat.col_names
        assert 'dec_deg' in randcat.col_names

        assert surr_ngal_mean > 0
        assert surr_ngal_rms < 5*surr_ngal_mean
        assert surr_ngal_mean + 3.01*surr_ngal_rms <= randcat.size
        assert 0 <= nksz <= 50
        
        self.box = box
        self.cosmo = cosmo
        self.randcat = randcat
        self.kbin_edges = core._check_kbin_edges(box, kbin_edges, use_dc)
        self.nkbins = len(kbin_edges)-1
        self.nksz = nksz
        self.surr_ngal_mean = surr_ngal_mean
        self.surr_ngal_rms = surr_ngal_rms
        self.nrand = randcat.size
        self.ztrue_col = ztrue_col
        self.zobs_col = zobs_col
        self.deltac = deltac
        self.kernel = kernel
        self.surr_ic_nbins = surr_ic_nbins
        self.spin0_hack = spin0_hack
        self.use_dc = use_dc

        ztrue = self._get_zcol(randcat, 'ztrue_col', ztrue_col)
        zobs = self._get_zcol(randcat, 'zobs_col', zobs_col)

        # Small optimization here: if (ztrue is zobs), then don't call ra_dec_to_xyz() twice.
        self.rcat_xyz_true = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=ztrue))
        self.rcat_xyz_obs = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=zobs)) if (ztrue is not zobs) else self.rcat_xyz_true
        
        fname = 'KszPSE2.__init__()'
        nrand = self.nrand

        # 1-d arrays of length self.nrand
        self.surr_bg = self._parse_gal_arg(surr_bg, fname, 'surr_bg', nrand, z=ztrue, non_negative=True, allow_none=False)
        self.rweights = self._parse_gal_arg(rweights, fname, 'rweights', nrand, z=zobs, non_negative=True, allow_none=True)

        # Length-nksz lists of (1-d array or None)
        self.ksz_bv = self._parse_ksz_arg(ksz_bv, fname, 'ksz_bv', nrand, z=ztrue, allow_none=False)
        self.ksz_rweights = self._parse_ksz_arg(ksz_rweights, fname, 'ksz_rweights', nrand, z=ztrue, allow_none=True)
        self.ksz_tcmb_realization = self._parse_tcmb_arg(ksz_tcmb_realization, fname, 'ksz_tcmb_realization', nrand)
        assert len(self.ksz_bv) ==  len(self.ksz_rweights) == len(self.ksz_tcmb_realization) == self.nksz
        
        # 1-d arrays of length self.nrand
        self.D = cosmo.D(z=ztrue, z0norm=True)
        self.faH = cosmo.frsd(z=ztrue) * cosmo.H(z=ztrue) / (1+ztrue)
        self.sigma2 = self._integrate_kgrid(self.box, cosmo.Plin_z0(self.box.get_k()))
        self.zobs = zobs   # needed for surr_ic_nbins

        # Check that we have enough randoms to make surrogate fields.
        self.bD_max = np.max(self.surr_bg * self.D)
        self.ngal_min = surr_ngal_mean - 3*surr_ngal_rms  # clamped to 3 sigma
        self.ngal_max = surr_ngal_mean + 3*surr_ngal_rms  # clamped to 3 sigma
        self.nrand_min = int(self.bD_max**2 * self.sigma2 * (self.ngal_max + 10))
        assert self.ngal_min >= 10
        
        if self.nrand < self.nrand_min:
            raise RuntimeError(f'KszPSE2: not enough randoms to make surrogate fields! This can be fixed by using'
                               + f' a larger random catalog (nrand={self.nrand}, nrand_min={self.nrand_min}).')

        # pse_rweights = Length (nksz+1) list of 1-d arrays.
        # win_footprints = Length (nksz+1) list of Fourier-space maps
        win_xyz = randcat.get_xyz(cosmo, zcol_name=zobs_col)  # not ztrue_col
        win_footprints = [ core.grid_points(box, win_xyz, self.rweights, kernel='cubic', fft=True, compensate=True, wscal = 1.0/np.sum(self.rweights)) ]
        pse_rweights = [ self.rweights ]

        for (w,bv) in zip(self.ksz_rweights, self.ksz_bv):
            wbv = (w*bv) if (w is not None) else bv
            wsum = np.sum(w) if (w is not None) else self.nrand
            win_footprints += [ core.grid_points(box, win_xyz, wbv, kernel='cubic', fft=True, compensate=True, wscal = 1.0/wsum) ]
            pse_rweights += [ wbv ]
        
        self.window_function = wfunc_utils.compute_wapprox(box, win_footprints)
        
        self.catalog_gridder = CatalogGridder(
            box = box,
            cosmo = cosmo,
            randcat = randcat,
            rweights = pse_rweights,
            nfootprints = nksz + 1,
            zcol_name = zobs_col,   # not ztrue_col
            save_rmaps = [True] + [False]*nksz,
            kernel = self.kernel
        )

    
    def eval_pk(self, gcat, gweights=None, ksz_gweights=None, ksz_bv=None, ksz_tcmb=None, zobs_col=None):
        r"""Computes $P_{gg}$, $P_{gv}$, and $P_{vv}$ from data or a mock (not a surrogate sim).

        Function args:

          - ``gcat`` (:class:`~kszx.Catalog`): galaxy catalog.

          - ``gweights`` (optional). Galaxy weighting $W_i^L$ used for the large-scale galaxy field $\delta_g(x)$.
             Can be specified as either:

              1. an array of length ``gcat.size``, to represent an arbitrary per-object $W_i^L$.
              2. a callable function $z \rightarrow f(z)$, if $W_i^L$ only depends on $z$.
              3. a scalar, if $W_i^L$ is the same for all galaxies.
              4. None (equivalent to ``gweights=1.0``).

            The ``gweights`` passed to ``eval_pk()`` should reflect as closely as possible the ``rweights``
            passed to the :class:`~kszx.KszPSE` constructor. For example, if randoms are FKP-weighted, then
            galaxies should also be FKP-weighted. (Warning: even a small mismatch between the weighting of
            galaxies/randoms can produce a large contribution to $P_{gg}$!)

          - ``ksz_gweights`` (optional): Galaxy weighting $W_i^S$ used for the kSZ velocity reconstruction $\hat v_r(x)$.
            Can be specified as either:

              1. an array of length ``randcat.size``, to represent an arbitrary per-object $W_i^S$.
              2. a callable function $z \rightarrow f(z)$, if $W_i^S$ only depends on $z$.
              3. a scalar, if $W_i^S$ is the same for all objects.
              4. None (equivalent to ``ksz_rweights=1``).
              5. a length-``nksz`` list (or iterable) of any of 1-4, if $W_i^S$ is not the same for all
                 velocity reconstructions being processed.

            The ``ksz_gweights`` passed to ``eval_pk()`` should reflect as closely as possible the 
            ``ksz_rweights`` passed to the :class:`~kszx.KszPSE` constructor.
    
          - ``ksz_bv``: Per-object KSZ velocity bias $b_v^i$. For more discsusion, and some key equations
            showing how to approximate $b_v$, see the ``ksz_bv`` argument in the :class:`~kszx.KszPSE`
            constructor docstring. Can be specified as either:

              1. an array of length ``gcat.size``, to represent an arbitrary per-object $b_v$.
              2. a callable function $z \rightarrow f(z)$, if $b_v$ only depends on $z$.
              3. a scalar, if $b_v$ is the same for all galaxies.
              4. a length-``nksz`` list (or iterable) of any of 1-3, if $b_v$ is not the same for all
                 velocity reconstructions bing processed. (This will usually be the case.)

            The ``ksz_bv`` argument passed to ``eval_pk()`` should reflect as closely as possible the 
            ``ksz_bv`` argument passed to the :class:`~kszx.KszPSE` constructor. (Same filtering, etc.)

          - ``ksz_tcmb``: 2-d array of shape ``(nksz, gcat.size)``.
            Note that each velocity reconstruction can use a different CMB filter. This is why $\tilde T$ is a 2-d
            array, indexed by ``0 <= i < nksz``, in addition to an index ``0 <= j < gcat.size``.

            The ``ksz_tcmb`` argument passed to ``eval_pk()`` should reflect as closely as possible the 
            ``ksz_tcmb_realization`` argument passed to the :class:`~kszx.KszPSE` constructor. (Same filtering, etc.)
        
          - ``zobs_col`` (string): name of column containing observed redshifts in ``gcat``.
            By default, we use the value ``zcol_obs`` that was specified when the constructor was called.

        Return value:

          - Array ``pk`` with shape ``(nksz+1, nksz+1, nkbins)``. The first two indices are
            field indices, with ordering (galaxy field, vrec fields). Thus the ``pk`` array
            contains all auto and cross power spectra $P_{gg}$, $P_{gv_i}$, $P_{v_iv_j}$.
        """

        assert isinstance(gcat, Catalog)

        if zobs_col is None:
            zobs_col = self.zobs_col
            
        z = self._get_zcol(gcat, 'zobs_col', zobs_col)        
        gcat_xyz = gcat.get_xyz(self.cosmo, zcol_name=zobs_col)
        print(f'{z.dtype =} {gcat_xyz.dtype = }')

        # Parse the 'gweights', 'ksz_gweights', 'ksz_bv', and 'ksz_tcmb' args.
        # (Same parsing logic as 'rweights', 'ksz_rweights', 'ksz_bv', and 'ksz_tcmb_realization' in __init__().)

        fname = 'KszPSE.eval_pk()'
        gweights = self._parse_gal_arg(gweights, fname, 'gweights', gcat.size, z, non_negative=True, allow_none=True)
        ksz_bv = self._parse_ksz_arg(ksz_bv, fname, 'ksz_bv', gcat.size, z, allow_none=False)
        ksz_gweights = self._parse_ksz_arg(ksz_gweights, fname, 'ksz_gweights', gcat.size, z, allow_none=True)
        ksz_tcmb = self._parse_tcmb_arg(ksz_tcmb, fname, 'ksz_tcmb', gcat.size)
        assert len(ksz_bv) == len(ksz_gweights) == len(ksz_tcmb) == self.nksz
        print(f'{gweights.dtype=} {ksz_bv.dtype=} {ksz_gweights.dtype=} {ksz_tcmb.dtype=}')

        # Initialize fmaps.

        gweights = gweights if (gweights is not None) else 1.0
        fmaps = [ self.catalog_gridder.grid_density_field(gcat, gweights, 0, zcol_name=zobs_col) ]

        for i in range(self.nksz):
            w, bv, t = ksz_gweights[i], ksz_bv[i], ksz_tcmb[i]
            print(f'{w.dtype=} {bv.dtype=} {t.dtype=}')
            coeffs = (w*t) if (w is not None) else t
            # FIXME mean subtraction now happens here (previously in Kpipe) -- need to make this less confusing
            print(f'{coeffs.dtype=}')
            coeffs = utils.subtract_binned_means(coeffs, z, nbins=25)
            print(f'{coeffs.dtype=}')
            wsum = np.dot(w,bv) if (w is not None) else np.sum(bv)
            print(f'{wsum.dtype=}')
            spin = 0 if self.spin0_hack else 1
            fmaps += [ self.catalog_gridder.grid_sampled_field(gcat, coeffs, wsum, i+1, spin=spin, zcol_name=zobs_col) ]

        # FIXME need some sort of CatalogPSE helper function here.
        pk = core.estimate_power_spectrum(self.box, fmaps, self.kbin_edges)
        pk *= np.reshape(self.catalog_gridder.ps_normalization, (self.nksz+1, self.nksz+1, 1))
        return pk
        
    
    def simulate_surrogate(self):
        r"""Makes a random surrogate simulation, and stores the result in class members (return value is None).
        
        $$\begin{align}
        S_g(x) &= \sum_{j\in\rm rand} S_g^j \delta^3(x-x_j) \\
        S_v(x) &= \sum_{j\in\rm rand} S_v^j \delta^3(x-x_j)
        \end{align}$$

        Initializes the following class members:

          - ``self.surr_ngal`` (integer): value of $N_{\rm gal}$ in the surrogate sim. (Recall that
            the value of $N_{\rm gal}$ in each surrogate sim is a Gaussian random variable. See
            ``surr_ngal_mean`` and ``surr_ngal_rms`` in the :class:`~kszx.KszPSE2` constructor
            docstring for more info.)

          - ``self.Sg_coeffs`` (1-d array of length ``self.nrand``): Coefficients $S_g^j$ above.

          - ``self.Sv_noise`` (shape-``(self.nksz, self.nrand)`` array): Coefficients $S_v^j$ above,
            contribution from reconstruction noise only.
        
          - ``self.Sv_signal`` (shape-``(self.nksz, self.nrand)`` array): Coefficients $S_v^j$ above,
            contribution from velocity field only (i.e. no reconstruction noise).

          - ``self.dSg_dfnl`` (1-d array of length ``self.nrand``): Derivative $dS_g^j/df_{NL}$.

        Note: Surrogates are factored into two functions (``simulate_surrogate()``and
        ``eval_pk_surrogate()`` so that the caller can put filtering logic in between, by
        operating directly on the ``self.Sg_coeffs``, ``self.Sv_noise``, ``Sv_signal``,
        and ``self.dSg_dfnl`` arrays. An example of filtering logic is subtracting the
        mean $\hat v_r$ in redshift bins, in order to mitigate foregrounds.
        """

        ngal = self.surr_ngal_mean + (self.surr_ngal_rms * np.random.normal())
        ngal = np.clip(ngal, self.ngal_min, self.ngal_max)
        ngal = int(ngal+0.5)  # round
        self.surr_ngal = ngal
        
        nrand = self.nrand
        assert 0 < ngal <= nrand
                
        delta0 = core.simulate_gaussian_field(self.box, self.cosmo.Plin_z0)
        core.apply_kernel_compensation(self.box, delta0, self.kernel)
        self.save_delta0 = delta0 

        # delta_g(k,z) = (bg + 2 fNL deltac (bg-1) / alpha(k,z)) * delta_m(k,z)
        #              = (bg D(z) + 2 fNL deltac (bg-1) / alpha0(k)) * delta0(k)

        Sg_prefactor = ((ngal/nrand) * self.rweights) if (self.rweights is not None) else (ngal/nrand)
        
        # deltaG = (bg * delta_m) = (bg * D * delta0), evaluated on randcat.
        bD = self.surr_bg * self.D
        delta_G = bD * core.interpolate_points(self.box, delta0, self.rcat_xyz_true, self.kernel, fft=True)
        self.Sg_coeffs = Sg_prefactor * delta_G

        eta_rms = np.sqrt((nrand/ngal) - (bD*bD) * self.sigma2)
        ug = np.random.normal(size = nrand)
        eta = ug * eta_rms
        self.Sg_coeffs += Sg_prefactor * eta
        self.save_ug = ug

        # Add term to deltag:
        #     2 fNL deltac (bg-1) / alpha(k,z) * delta_m(k,z)
        #   = 2 fNL deltac (bg-1) / alpha_z0(k) * delta0(k)    [ factor D(z) cancels ]
        phi0 = core.multiply_kfunc(self.box, delta0, lambda k: 1.0/self.cosmo.alpha_z0(k=k), dc=0)
        phi0 = core.interpolate_points(self.box, phi0, self.rcat_xyz_true, self.kernel, fft=True)
        self.dSg_dfnl = Sg_prefactor * (2 * self.deltac) * (self.surr_bg-1) * phi0

        if self.surr_ic_nbins > 0:
            self.Sg_coeffs[:] = utils.subtract_binned_means(self.Sg_coeffs, self.zobs, self.surr_ic_nbins)
            self.dSg_dfnl[:] = utils.subtract_binned_means(self.dSg_dfnl, self.zobs, self.surr_ic_nbins)
        
        self.Sv_noise = np.zeros((self.nksz, self.nrand))
        self.Sv_signal = np.zeros((self.nksz, self.nrand))
        
        if self.nksz == 0:
            return

        # vr = (faHD/k) delta0, evaluated at rcat_xyz_true.
        vr = core.multiply_kfunc(self.box, delta0, lambda k: 1.0/k, dc=0)
        vr = core.interpolate_points(self.box, vr, self.rcat_xyz_true, self.kernel, fft=True, spin=1)
        vr *= self.faH
        vr *= self.D

        # M = vector with (nrand-ngal) 0s and (ngal) 1s, in random positions.
        # (This way of generating M is a little slow, but I don't think there's a faster way to do
        # it in numpy, and it's not currently a bottleneck.)
        M = np.zeros(nrand)
        M[:ngal] = 1.0
        M = np.random.permutation(M)
        self.save_M = M
        
        for i in range(self.nksz):
            self.Sv_noise[i,:] = M * self.ksz_tcmb_realization[i]
            self.Sv_signal[i,:] = (ngal/nrand) * self.ksz_bv[i] * vr

            if self.ksz_rweights[i] is not None:
                self.Sv_noise[i,:] *= self.ksz_rweights[i]
                self.Sv_signal[i,:] *= self.ksz_rweights[i]


    def eval_pk_surrogate(self):
        r"""Returns an array of shape (2*nksz+2, 2*nksz+2, nkbins), obtained from the current surrogate sim.

        The first two indices of the array are "field indices" as follows:
            0 = surrogate sim S_g, with fNL=0.
            1 = surrogate dS_g/dfNL
            (2*i+2) = surrogate sim S_v, reconstruction noise contribution only (where 0 <= i < nksz).
            (2*i+3) = surrogate sim S_v, velocity signal contribution only (where 0 <= i < nksz).

        Thus, the returned shape (2*nksz+2, 2*nksz+2, nkbins) array can be used to compute P_gg, P_gv, P_vv
        for all combinations of CMB frequencies, and arbitrary (fNL, bv).

        Reminder: logic for surrogate sims is split between :meth:`simulate_surrogate()` and this method.
        When :meth:`simulate_surrogate()` is called, it initializes coefficient arrays (``self.Sg_coeffs`` etc.)
        which are used here to simulate surrogate fields and estimate power spectra.
        """

        if not hasattr(self, 'Sg_coeffs'):
            raise RuntimeError("Before calling KszPSE.eval_pk_surrogate(), you must call KszPSE.simulate_surrogate()")
        
        Nr = self.nrand
        Ng = self.surr_ngal
        Sg_wsum = ((Ng/Nr) * np.sum(self.rweights)) if (self.rweights is not None) else Ng

        # fmaps = list of Fourier-space maps
        # fmaps[0] = surrogate sim S_g, with fNL=0.
        # fmaps[1] = surrogate dS_g/dfNL
        # fmaps[2*i+2] = surrogate sim S_v, reconstruction noise contribution only (where 0 <= i < nksz).
        # fmaps[2*i+3] = surrogate sim S_v, velocity signal contribution only (where 0 <= i < nksz).
        
        fmaps = [ ]
        fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.Sg_coeffs, Sg_wsum, 0, zcol_name=self.zobs_col) ]
        fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.dSg_dfnl, Sg_wsum, 0, zcol_name=self.zobs_col) ]
        
        for i in range(self.nksz):
            w, bv = self.ksz_rweights[i], self.ksz_bv[i]
            Sv_wsum = (Ng/Nr) * (np.dot(w,bv) if (w is not None) else np.sum(bv))
            spin = 0 if self.spin0_hack else 1
            fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.Sv_noise[i,:], Sv_wsum, i+1, spin=spin, zcol_name=self.zobs_col) ]
            fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.Sv_signal[i,:], Sv_wsum, i+1, spin=spin, zcol_name=self.zobs_col) ]

        # Unnormalized power spectrum estimates (from CatalogGridder)
        pk = core.estimate_power_spectrum(self.box, fmaps, self.kbin_edges)
        assert pk.shape == (2*self.nksz+2, 2*self.nksz+2, self.nkbins)

        # Index mapping for power spectrum normalization.
        imap = [ (i//2) for i in range(2*self.nksz+2) ]

        # Apply power spectrum normalization
        for i in range(2*self.nksz+2):
            for j in range(2*self.nksz+2):
                pk[i,j,:] *= self.catalog_gridder.ps_normalization[imap[i],imap[j]]

        return pk


    @classmethod
    def reduce_pk(cls, pk, fnl, bv):
        """Reduces the (2*nksz+2, 2*nksz+2, nkbins)  array returned by eval_pk_surrogate() by fixing fNL and bv.

        The input array 'pk' can either be
          - an array of shape (2*nksz+2, 2*nksz+2, nkbins) returned by eval_pk_surrogate(), representing one sim
          - an array of shape (nsurr, 2*nksz+2, 2*nksz+2, nkbins) representing many sims

        The output array has shape
          - (nksz+1, nksz+1, nkbins) in the first case above
          - (nsurr, nksz+1, nksz+1, nkbins) in the second case
        """

        pk = np.asarray(pk)
        
        if not cls._valid_shape_for_reduce_pk(pk):
            raise RuntimeError(f"KszPSE2.reduce_pk(): 'pk' argument has invalid shape {pk.shape}")

        nksz = (pk.shape[1]//2) - 1
        coeffs = np.array([fnl] + [bv]*nksz)

        # Reduce axis 0
        if pk.ndim == 3:
            pk = pk[::2] + pk[1::2] * coeffs.reshape((nksz+1,) + (1,)*(pk.ndim-1))

        # Reduce axis 1
        pk = pk[:,::2] + pk[:,1::2] * coeffs.reshape((1,nksz+1) + (1,)*(pk.ndim-2))

        # Reduce axis 2
        if pk.ndim == 4:
            pk = pk[:,:,::2] + pk[:,:,1::2] * coeffs.reshape((1,1,nksz+1) + (1,)*(pk.ndim-3))

        return pk
    
        
    @classmethod
    def _valid_shape_for_reduce_pk(cls, pk):
        if (pk.ndim < 3) or (pk.ndim > 4):
            return False
        if pk.shape[pk.ndim-3] != pk.shape[pk.ndim-2]:
            return False
        if (pk.shape[1] % 2) or (pk.shape[1] < 4):
            return False
        return True
        
        
    ####################################################################################################


    def _get_zcol(self, catalog, zcol_argname, zcol_name):
        """Helper method called by constructor, to check that column exists in catalog."""

        if not isinstance(zcol_name, str):
            raise RuntimeError(f"KszPSE2: expected argument {zcol_argname}={zcol_name} to be a string")

        if zcol_name not in catalog.col_names:
            raise RuntimeError(f"KszPSE2: catalog does not contain column '{zcol_name}' (specified as argument '{zcol_argname}')")

        z = getattr(catalog, zcol_name)
        zmin = np.min(z)
        
        if zmin <= 0:
            raise RuntimeError(f"KszPSE2: expected all redshifts to be positive (got min(catalog.{zcol_name}) = {zmin})")

        return z

    
    def _parse_gal_arg(self, f, funcname, argname, ngal, z, allow_none=False, non_negative=False):
        """Used to parse constructor args 'surr_bg' and 'rweights'.

        The 'f' argument is either:
           1. a callable function z -> f(z)
           2. an array of length ngal
           3. a scalar
           4. None (equivalent to 1.0)

        If 'z' is None, then a function (option #1 above) is not allowed.
        If 'allow_none' is False, then None (option #4 above) is not allowed.

        Returns an array of length ngal.
        """

        if f is None:
            if allow_none:
                return np.full(ngal, 1.0)
            raise RuntimeError(f"{funcname}: '{argname}' argument must be specified")
        
        if callable(f):
            if z is None:
                raise RuntimeError(f"{funcname}: '{argname}' constructor argument is a function -- this is not currently allowed")
            fz = f(z)
            fz_name = f'return value from {argname}()'
            fz = utils.asarray(fz, funcname, fz_name, dtype=float)
            if fz.shape != (ngal,):
                raise RuntimeError(f"{fz_name} has shape {fz.shape}, expected shape {(ngal,)}")
        
        else:
            fz = utils.asarray(f, funcname, argname, dtype=float)
        
            if fz.ndim == 0:
                fz = np.full(ngal, fz)
            elif fz.shape != (ngal,):
                raise RuntimeError(f"{funcname}: expected '{argname}' constructor arg to be either"
                                   + f" shape ({ngal},) or a scalar (got shape {fz.shape})")

        assert fz.shape == (ngal,)
        assert fz.dtype == float

        if non_negative and np.min(fz) < 0:
            raise RuntimeError(f"{funcname}: expected {argname} to be >= 0")

        return fz


    def _parse_ksz_arg(self, f, funcname, argname, ngal, z, allow_none=False, non_negative=False):
        """
        Used to parse constructor args 'ksz_rweights', 'ksz_bv', 'ksz_tcmb_realization'.

        The 'f' argument is either:
           1. a callable function z -> f(z)
           2. an array of length ngal
           3. a scalar
           4. None (equivalent to 1.0)
           5. a length-nksz list (or iterable) of any of 1-4.

        If 'z' is None, then a function (option #1 above) is not allowed.
        If 'allow_none' is False, then None (option #4 above) is not allowed.
        Returns a length-nksz list of (1-d numpy array or None), where 1-d numpy arrays have length ngal.
        """

        if self.nksz == 0:
            # All kSZ-related constructor args must be either None, or an empty iterable.
            if (f is None) or self._is_empty_iterable(f):
                return []
            raise RuntimeError(f"{funcname}: argument '{argname}' was specified, but 'nksz' constructor arg was unspecified (or zero)")

        is_small_array = isinstance(f, np.ndarray) and (f.shape[0] != self.nksz)

        # These cases can be handled by a single call to self._parse_gal_arg().
        if is_small_array or (f is None) or callable(f) or self._is_scalar(f):
            x = self._parse_gal_arg(f, funcname, argname, ngal, z, allow_none=allow_none, non_negative=non_negative)
            return [x] * self.nksz

        try:
            flen = len(f)
        except:
            raise RuntimeError(f"{funcname}: couldn't parse argument '{argname}' (got type {type(f)},"
                               + " expected iterable, function, scalar, or None)")

        if flen != self.nksz:
            raise RuntimeError(f"{funcname}: couldn't parse argument '{argname}' (got length-{flen} iterable non-array,"
                               + f" expected array, function, scalar, None, or iterable with length nksz={self.nksz})")

        # This case can be handled by multiple calls to self._parse_gal_arg().
        return [ self._parse_gal_arg(x, funcname, f'{argname}[{i}]', ngal, z, allow_none=allow_none, non_negative=non_negative) for i,x in enumerate(f) ]

    
    def _parse_tcmb_arg(self, x, funcname, argname, ngal):
        """Used to parse 'ksz_tcmb_realization' constructor arg, or 'ksz_tcmb' arg to eval_pk()."""
        
        if self.nksz == 0:
            if (x is None) or self._is_empty_array(f):
                return np.zeros((self.nksz,ngal), dtype=float)
            raise RuntimeError(f"{funcname}: argument '{argname}' was specified, but 'nksz' constructor arg was unspecified (or zero)")

        x = utils.asarray(x, funcname, argname, dtype=float)

        if x.shape == (self.nksz, ngal):
            return x
        
        raise RuntimeError(f"{funcname}: expected argument '{argname}' to have shape (nksz,ngal)={(self.nksz,ngal)}, got shape {x.shape}")

    
    @staticmethod
    def _is_empty_iterable(f):
        try:
            for _ in f:
                return False
            return True
        except:
            pass
        return False

    
    @staticmethod
    def _is_empty_array(x):
        try:
            if np.asarray(x).size == 0:
                return True
        except:
            pass
        return False
    
    
    @staticmethod
    def _is_scalar(f):
        try:
            f = float(f)
            return True
        except:
            pass
        return False

    
    @staticmethod
    def _integrate_kgrid(box, kgrid):
        """Helper method for constructor. Could be moved somewhere more general (e.g. kszx.core)."""

        assert kgrid.shape == box.fourier_space_shape
    
        w = np.full(box.fourier_space_shape[-1], 2.0)
        w[0] = 1.0
        w[-1] = 2.0 if (box.real_space_shape[-1] % 2) else 1.0
        w = np.reshape(w, (1,)*(box.ndim-1) + (-1,))

        # FIXME could be optimized
        ret = np.sum(w * kgrid) / box.box_volume
        return np.real(ret)


####################################################################################################


class KszPipe:
    def __init__(self, input_dir, output_dir, nsurr=400):
        """KSZ P(k) pipeline.

        Runnable either by calling the run() method, or from the command line with:
           python -m kszx <input_dir> <output_dir>

        Reminder: input directory must contain files { params.yml, galaxies.h5, randoms.h5,
        bounding_box.pkl }. The pipeline will populate the output directory with files
        { params.yml, pk_data.npy, pk_surrogates.npy }. (For details of these file formats,
        see one of the README files that Kendrick or Selim have lying around.)
        """
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.nsurr = nsurr
        
        # Create output_dir and 'tmp' subdir
        os.makedirs(f'{output_dir}/tmp', exist_ok=True)

        with open(f'{input_dir}/params.yml', 'r') as f:
            params = yaml.safe_load(f)

            self.surr_bg = params['surr_bg']
            self.surr_ic_nbins = params['surr_ic_nbins']
            self.spin0_hack = params['spin0_hack']
            assert self.spin0_hack in [ True, False ]

            self.kmax = params['kmax']
            self.nkbins = params['nkbins']
            self.kbin_edges = np.linspace(0, self.kmax, self.nkbins+1)
            self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.

        self.box = io_utils.read_pickle(f'{input_dir}/bounding_box.pkl')
        self.kernel = 'cubic'
        self.deltac = 1.68

        # This code moved into cached properties (see below), in order to reduce startup time.
        # self.cosmo = Cosmology('planck18+bao')
        # self.gcat = Catalog.from_h5(f'{input_dir}/galaxies.h5')        
        # self.rcat = Catalog.from_h5(f'{input_dir}/randoms.h5')

        self.pk_data_filename = f'{output_dir}/pk_data.npy'
        self.pk_surr_filename = f'{output_dir}/pk_surrogates.npy'
        self.pk_single_surr_filenames = [ f'{output_dir}/tmp/pk_surr_{i}.npy' for i in range(nsurr) ]


    @functools.cached_property
    def cosmo(self):
        return Cosmology('planck18+bao')

    @functools.cached_property
    def gcat(self):
        return Catalog.from_h5(f'{self.input_dir}/galaxies.h5')

    @functools.cached_property
    def rcat(self):
        return Catalog.from_h5(f'{self.input_dir}/randoms.h5')

    @functools.cached_property
    def surrogate_factory(self):
        # FIXME needs comment
        surr_ngal_mean = self.gcat.size
        surr_ngal_rms = 4 * np.sqrt(self.gcat.size)  # 4x Poisson
        return SurrogateFactory(self.box, self.cosmo, self.rcat, surr_ngal_mean, surr_ngal_rms, 'ztrue')
    
    @functools.cached_property
    def pse(self):
        """Returns a KszPSE2 object."""
        
        print('Initializing KszPSE2: this will take a few minutes')
        
        # FIXME needs comment
        surr_ngal_mean = self.gcat.size
        surr_ngal_rms = 4 * np.sqrt(self.gcat.size)  # 4x Poisson
        
        rweights = getattr(self.rcat, 'weight_zerr', None)
        vweights = getattr(self.rcat, 'vweight_zerr', None)
        
        pse = KszPSE2(
            box = self.box, 
            cosmo = self.cosmo, 
            randcat = self.rcat, 
            kbin_edges = self.kbin_edges,
            surr_ngal_mean = surr_ngal_mean,
            surr_ngal_rms = surr_ngal_rms,
            surr_bg = self.surr_bg,
            rweights = rweights,
            nksz = 2,
            ksz_rweights = vweights,
            ksz_bv = [ self.rcat.bv_90, self.rcat.bv_150 ],
            ksz_tcmb_realization = [ self.rcat.tcmb_90, self.rcat.tcmb_150 ],
            ztrue_col = 'ztrue',
            zobs_col = 'zobs',
            surr_ic_nbins = self.surr_ic_nbins,
            spin0_hack = self.spin0_hack
        )
        
        print('KszPSE2 initialization done')
        return pse

    
    def get_pk_data(self, run=False):
        """Returns a shape (3,3,nkbins) array.

        If run=False, then this function expects the P(k) file to be on disk from a previous pipeline run.
        If run=True, then the P(k) file will be computed if it is not on disk.
        """

        if os.path.exists(self.pk_data_filename):
            return io_utils.read_npy(self.pk_data_filename)
        
        if not run:
            raise RuntimeError(f'KszPipe.get_pk_data(): run=False was specified, and file {self.pk_data_filename} not found')

        print('get_pk_data(): running\n', end='')

        # FIXME mean subtraction moved into KszPSE2 -- need to make this less confusing
        # t90 = utils.subtract_binned_means(self.gcat.tcmb_90, self.gcat.z, nbins=25)
        # t150 = utils.subtract_binned_means(self.gcat.tcmb_150, self.gcat.z, nbins=25)

        gweights = getattr(self.gcat, 'weight_zerr', None)
        vweights = getattr(self.gcat, 'vweight_zerr', None)

        pk_data = self.pse.eval_pk(
            gcat = self.gcat,
            gweights = gweights,
            ksz_gweights = vweights,
            ksz_bv = [ self.gcat.bv_90, self.gcat.bv_150 ], 
            ksz_tcmb = [ self.gcat.tcmb_90, self.gcat.tcmb_150 ],
            zobs_col = 'z'
        )

        io_utils.write_npy(self.pk_data_filename, pk_data)
        return pk_data

    
    def get_pk_data2(self, run=False, force=False):
        gweights = getattr(self.gcat, 'weight_zerr', np.ones(self.gcat.size))
        rweights = getattr(self.rcat, 'weight_zerr', np.ones(self.rcat.size))
        vweights = getattr(self.gcat, 'vweight_zerr', np.ones(self.gcat.size))
        print(f'{gweights.dtype =}')
        print(f'{rweights.dtype =}')
        print(f'{vweights.dtype =}')
        
        gcat_xyz = self.gcat.get_xyz(self.cosmo)
        rcat_xyz = self.rcat.get_xyz(self.cosmo, zcol_name='zobs')  # not ztrue
        print(f'{gcat_xyz.dtype = }')
        print(f'{rcat_xyz.dtype = }')

        bv_cols = [ self.gcat.bv_90, self.gcat.bv_150 ]
        tcmb_cols = [ self.gcat.tcmb_90, self.gcat.tcmb_150 ]

        fmaps = [ core.grid_points(self.box, gcat_xyz, gweights, rcat_xyz, rweights, kernel=self.kernel, fft=True, compensate=True) ]
        weights = [ np.sum(gweights) ]  # reminder: footprints are normalized to sum(weights)=1

        for bv, tcmb in zip(bv_cols, tcmb_cols):
            print(f'{bv.dtype =} {tcmb.dtype =}')
            coeffs = vweights * tcmb
            coeffs = utils.subtract_binned_means(coeffs, self.gcat.z, nbins=25)  # note mean subtraction here
            print(f'{coeffs.dtype =}')
            fmaps += [ core.grid_points(self.box, gcat_xyz, coeffs, kernel=self.kernel, fft=True, spin=1, compensate=True) ]
            weights += [ np.sum(vweights) ]
        
        wf = wfunc_utils.scale_wapprox(self.pse.window_function, weights)
        print(f'{wf.dtype =}')
        pk = core.estimate_power_spectrum(self.box, fmaps, self.kbin_edges)
        print(f'{pk.dtype =}')
        pk /= wf[:,:,None]
        print(f'{pk.dtype =}')
        return pk

        
    def get_pk_surrogate(self, isurr, run=False):
        """Returns a shape (6,6,nkbins) array.
        
        If run=False, then this function expects the P(k) file to be on disk from a previous pipeline run.
        If run=True, then the P(k) file will be computed if it is not on disk.
        """
        
        fname = self.pk_single_surr_filenames[isurr]
        
        if os.path.exists(fname):
            return io_utils.read_npy(fname)

        if not run:
            raise RuntimeError(f'KszPipe.get_pk_surrogate(): run=False was specified, and file {fname} not found')

        print(f'get_pk_surrogate({isurr}): running\n', end='')
        
        self.pse.simulate_surrogate()
        
        for sv in [ self.pse.Sv_noise, self.pse.Sv_signal ]:
            assert sv.shape ==  (2, self.rcat.size)
            for j in range(2):
                sv[j,:] = utils.subtract_binned_means(sv[j,:], self.rcat.zobs, nbins=25)
    
        pk = self.pse.eval_pk_surrogate()

        io_utils.write_npy(fname, pk)
        return pk


    def get_pk_surrogates(self):
        """Returns a shape (nsurr,6,6,nkins) array, containing P(k) for all surrogates."""
        
        if os.path.exists(self.pk_surr_filename):
            return io_utils.read_npy(self.pk_surr_filename)

        if not all(os.path.exists(f) for f in self.pk_single_surr_filenames):
            raise RuntimeError(f'KszPipe.read_pk_surrogates(): necessary files do not exist; you need to call KszPipe.run()')

        pk = np.array([ io_utils.read_npy(f) for f in self.pk_single_surr_filenames ])
        
        io_utils.write_npy(self.pk_surr_filename, pk)
        return pk

    
    def get_pk_surrogate2(self, ngal=None, delta=None, M=None, ug=None):        
        nfreq = 2
        zobs = self.rcat.zobs
        nrand = self.rcat.size
        bv_list = [ self.rcat.bv_90, self.rcat.bv_150 ]
        tcmb_list = [ self.rcat.tcmb_90, self.rcat.tcmb_150 ]
        rweights = getattr(self.rcat, 'weight_zerr', np.ones(nrand))
        vweights = getattr(self.rcat, 'vweight_zerr', np.ones(nrand))
        xyz_obs = self.rcat.get_xyz(self.cosmo, 'zobs')

        if ug is None:
            ug = np.random.normal(size=nrand)
            
        self.surrogate_factory.simulate_surrogate(ngal=ngal, delta=delta, M=M)

        ngal = self.surrogate_factory.ngal
        bD = self.surr_bg * self.surrogate_factory.D
        eta_rms = np.sqrt((nrand/ngal) - (bD*bD) * self.surrogate_factory.sigma2)
        eta = ug * eta_rms

        # Coeffs
        Sg = (ngal/nrand) * rweights * (self.surr_bg * self.surrogate_factory.delta + eta)
        dSg_dfnl = (ngal/nrand) * rweights * (2 * self.deltac) * (self.surr_bg-1) * self.surrogate_factory.phi        
        Sv_noise = np.zeros((nfreq, nrand))
        Sv_signal = np.zeros((nfreq, nrand))

        for i,(bv,tcmb) in enumerate(zip(bv_list,tcmb_list)):
            Sv_noise[i,:] = vweights * self.surrogate_factory.M * tcmb
            Sv_signal[i,:] = (ngal/nrand) * vweights * bv * self.surrogate_factory.vr

        Sg = utils.subtract_binned_means(Sg , zobs, self.surr_ic_nbins)
        dSg_dfnl = utils.subtract_binned_means(dSg_dfnl, zobs, self.surr_ic_nbins)

        for sv in [ Sv_noise, Sv_signal ]:
            assert sv.shape ==  (2, nrand)
            for j in range(2):
                sv[j,:] = utils.subtract_binned_means(sv[j,:], zobs, nbins=25)

        # Kpipe -> KszPSE
        
        fmaps_new = [ ]
        weights_new = [ ]
        Sg_wsum = ngal * np.mean(rweights)
        
        fmaps_new += [ core.grid_points(self.box, xyz_obs, Sg, kernel=self.kernel, fft=True, spin=0, compensate=True) ]
        fmaps_new += [ core.grid_points(self.box, xyz_obs, dSg_dfnl, kernel=self.kernel, fft=True, spin=0, compensate=True) ]
        weights_new += [ Sg_wsum, Sg_wsum ]

        for i,(bv,tcmb) in enumerate(zip(bv_list,tcmb_list)):
            Sv_wsum = (ngal/nrand) * np.dot(vweights,bv)
            fmaps_new += [ core.grid_points(self.box, xyz_obs, Sv_noise[i,:], kernel=self.kernel, fft=True, spin=1, compensate=True) ]
            fmaps_new += [ core.grid_points(self.box, xyz_obs, Sv_signal[i,:], kernel=self.kernel, fft=True, spin=1, compensate=True) ]
            wnew = ngal * np.mean(vweights)
            weights_new += [ wnew, wnew ]

        wf = wfunc_utils.scale_wapprox(self.pse.window_function, weights_new, [0,0,1,1,2,2])
        pk_new = core.estimate_power_spectrum(self.box, fmaps_new, self.kbin_edges)
        pk_new /= wf[:,:,None]
        return pk_new

    
    def run(self, processes=2):
        """Runs pipeline. If any output files already exist, they will be skipped."""
        
        # Copy yaml file from input to output dir.
        if not os.path.exists(f'{self.output_dir}/params.yml'):
            shutil.copyfile(f'{self.input_dir}/params.yml', f'{self.output_dir}/params.yml')
                
        have_data = os.path.exists(self.pk_data_filename)
        have_surr = os.path.exists(self.pk_surr_filename)
        missing_surrs = [ ] if have_surr else [ i for (i,f) in enumerate(self.pk_single_surr_filenames) if not os.path.exists(f) ]

        if (not have_surr) and (len(missing_surrs) == 0):
            self.get_pk_surrogates()   # creates "top-level" file
            have_surr = True
            
        if have_data and have_surr:
            print(f'KszPipe.run(): pipeline has already been run, exiting early')
            return
        
        # Initialize KszPSE2 before creating multiprocessing Pool.
        self.pse

        # FIXME currently I don't have a good way of setting the number of processes automatically --
        # caller must adjust the number of processes to the amount of memory in the node.
        
        with utils.Pool(processes) as pool:
            l = [ ]
                
            if not have_data:
                l += [ pool.apply_async(self.get_pk_data, (True,)) ]
            for i in missing_surrs:
                l += [ pool.apply_async(self.get_pk_surrogate, (i,True)) ]

            for x in l:
                x.get()

        if not have_surr:
            # Consolidates all surrogates into one file
            self.get_pk_surrogates()
