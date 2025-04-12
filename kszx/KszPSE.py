import numpy as np

from .Box import Box
from .Catalog import Catalog
from .Cosmology import Cosmology
from .CatalogGridder import CatalogGridder

from . import utils
from . import core


class KszPSE:
    def __init__(self, box, cosmo, randcat, kbin_edges, surr_ngal_mean, surr_ngal_rms, surr_bg, rweights=None, nksz=0, ksz_rweights=None, ksz_bv=None, ksz_tcmb_realization=None, ztrue_col='z', zobs_col='z', deltac=1.68, kernel='cubic', surr_ic=True, use_dc=False):
        r"""KszPSE ("KSZ power spectrum estimator"): a high-level pipeline class for $P_{gg}$, $P_{gv}$, and $P_{vv}$.

        Features:

          - Processes multiple velocity reconstructions efficiently in parallel -- for example,
            separate velocity reconstructions for 90 and 150 GHz. This feature can also be used
            to efficiently explore multiple possibilities for the foreground mask, CMB filtering,
            etc, by defining more velocity reconstructions.

          - Runs "surrogate" sims (see overleaf) to characterize the survey window function,
            and assign error bars to power spectra.
 
          - Surrogate sims allow $f_{NL} \ne 0$, to characterize effect of $f_{NL}$ on windowed power 
            spectra, down to arbitrarily low $k$.

          - Velocity reconstruction noise is included in surrogate sims via a bootstrap procedure,
            using the observed CMB realization. This automatically incorporates noise inhomogeneity
            and "striping", and captures correlations e.g. between 90 and 150 GHz.

          - The galaxy catalog can be spectroscopic or photometric (via the ``ztrue_col`` and 
            ``zobs_col`` constructor args). Surrogate sims will capture the effect of photo-z errors.

          - The windowed power spectra $P_{gg}$, $P_{gv}$, $P_{vv}$ use a normalization which
            should be approximately correct. The normalization is an ansatz which is imperfect,
            especially on large scales, so surrogate sims should still be used to compare power
            spetra to models. (However, it's convenient to have a normalization which is not
            "off" by orders of magnitude!) Eventually, we'll implement precise window function
            deconvolution.
        
            Power spectra also include "compensation factors" (see :func:`~kszx.apply_kernel_compensation()`)
            to mitigate interpolation/gridding biases at high $k$.

          - We currently assume only one galaxy field (and only one random field) for simplicity,
            whereas we allow multiple velocity reconstructions. This could be generalized if needed.

        Usage:

          - Construct a KszPSE object (see constructor syntax below).

          - To evaluate power spectra ($P_{gg}$, $P_{gv}$, or $P_{vv}$) on **data or a mock**, 
            call :meth:`KszPSE.eval_pk()` with a galaxy catalog (``gcat``), and filtered CMB 
            temperatures at the galaxy locations (``ksz_tmcb``).

          - To evaluate power spectra on a **surrogate simulation**, call 
            :meth:`KszPSE.simulate_surrogate()` to make the simulation, then call
            :meth:`KszPSE.eval_pk_surrogate()` to estimate power spectra.
        
        Note: Surrogates are factored into two functions (``simulate_surrogate()``and
        ``eval_pk_surrogate()`` so that the caller can put filtering logic in between, by
        operating directly on the ``self.Sg_coeffs``, ``self.Sv_noise``, ``Sv_signal``,
        and ``self.dSg_dfNL`` arrays (see :meth:`KszPSE.simulate_surrogate()` docstring).
        An example of filtering logic is subtracting the mean $\hat v_r$ in redshift bins,
        in order to mitigate foregrounds.
        
        An example notebook where KszPSE is used (but not until later in the notebook):

           https://github.com/kmsmith137/kszx_notebooks/blob/main/05_sdss_pipeline/05_exploratory_plots.ipynb

        The KszPSE computes power spectra involving a galaxy density field $\rho_g$,
        one or more kSZ velocity reconstructions $\hat v_r$, and surrogate fields $S_g, S_v$.
        Definitions of these fields are given in the overleaf, and can be summarized as follows:

        $$\begin{align}
        \rho_g(x) &= \bigg( \sum_{i\in \rm gal} W_i^L \, \delta^3(x-x_i) \bigg) - \frac{N_g}{N_r} \bigg( \sum_{j\in \rm rand} W_j^L \, \delta^3(x-x_j) \bigg) \\
        \hat v_r(x) &= \sum_{i\in \rm gal} W_i^S \, \tilde T(\theta_i) \, \delta^3(x-x_i) \\
        S_g(x) &= \sum_{j\in \rm rand} \frac{N_g}{N_r} W_j^L \big( b_j^G \delta_m(x_j) + \eta_j \big) \delta^3(x-x_j) \\
        S_v(x) &= \sum_{j\in\rm rand} \bigg( \frac{N_g}{N_r} W_j^S b_j^v v_r(x_j) + M_j W_j^S \tilde T(\theta_j) \bigg)  \delta^3(x-x_j)
        \end{align}$$
        For more details, and a formal specification of what KszPSE computes, see ":ref:`ksz_pse_details`" 
        in the sphinx docs.

        Constructor args:

          - ``box`` (:class:`~kszx.Box`): defines pixel size, bounding box size, and location of observer.

          - ``cosmo`` (:class:`~kszx.Cosmology`).

          - ``randcat`` (:class:`~kszx.Catalog`): random catalog, defines survey footprint and redshift
            distribution. The randcat must contain columns ``ra_deg`` and ``dec_deg``.

            **Note:** By default, the galaxy survey is assumed spectroscopic, with a single redshift
            column ``z``. If the catalog is photometric (or if the redshift column is not named ``z``),
            then you'll want to specify the ``ztrue_col`` and ``zobs_col`` constructor args (see below).

          - ``kbin_edges`` (array): 1-d array of length (nkbins+1) defining $k$-bin endpoints for
            power spectrum estimation. The i-th bin covers k-range ``kbin_edges[i] <= i < kbin_edges[i+1]``.

          - ``surr_ngal_mean`` and ``surr_ngal_rms`` (float): In the surrogate sims, I decided to allow $N_{\rm gal}$
            to vary from one surrogate sim to the next. (The idea is to make the surrogate sims more similar to
            mocks, where $N_{\rm gal}$ varies between mocks. Indeed, I find that allowing $N_{\rm gal}$ to vary
            in the surrogate sims does improve overall agreement with mocks.)

            In each surrogate sim, $N_{\rm gal}$ is a Gaussian random variable with mean/rms given by
            the ``surr_ngal_mean`` and ``surr_ngal_rms`` constructor args. If mocks are available, then 
            one way to get sensible values for these arguments is to use the mean/variance in the mocks.
            As a simple placeholder, you could also take ``surr_ngal_rms=0`` (to disable varying $N_{\rm gal}$
            entirely) or ``surr_ngal_rms = sqrt(surr_ngal_mean)`` (Poisson statistics).

          - ``surr_bg``: Galaxy bias $b_g$ used in surrogate sims. Can be specified as either:

               1. an array of length ``randcat.size``, to represent an arbitrary per-object $b_g$.
               2. a callable function $z \rightarrow f(z)$, if $b_g$ only depends on $z$.
               3. a scalar, if $b_g$ is the same for all galaxies.

          - ``rweights`` (optional): Galaxy weighting $W_i^L$ used for the large-scale galaxy field $\delta_g(x)$
            (see equations earlier in this docstring, or ":ref:`ksz_pse_details`" in the sphinx docs).
            Can be specified as either:

               1. an array of length ``randcat.size``, to represent an arbitrary per-object $W_i^L$.
               2. a callable function $z \rightarrow f(z)$, if $W_i^L$ only depends on $z$.
               3. a scalar, if $W_i^L$ is the same for all galaxies.
               4. None (equivalent to ``rweights=1.0``).

            **Note:** there is also a galaxy weighting $W_i^S$ used in the kSZ velocity reconstruction $\hat v_r(x)$.
            The weighting $W_i^S$ is specified as the ``ksz_rweights`` constructor arg, see below.
            The weightings $W_i^L$ and $W_i^S$ are independent.

          - ``nksz`` (integer): Number of kSZ velocity reconstruction fields $\hat v_r(x)$.
            For example, to separate velocity reconstructions from 90 and 150 GHz, use ``nksz=2``.
            If ``nksz`` is zero or unspecified, then the KszPSE will estimate the power spectrum of 
            a galaxy field $\delta_g$, but no $\hat v_r$ fields.

          - ``ksz_rweights`` (optional): Galaxy weighting $W_i^S$ used for the kSZ velocity reconstruction $\hat v_r(x)$.
            Can be specified as either:

              1. an array of length ``randcat.size``, to represent an arbitrary per-object $W_i^S$.
              2. a callable function $z \rightarrow f(z)$, if $W_i^S$ only depends on $z$.
              3. a scalar, if $W_i^S$ is the same for all objects.
              4. None (equivalent to ``ksz_rweights=1``).
              5. a length-``nksz`` list (or iterable) of any of 1-4, if $W_i^S$ is not the same for all
                 velocity reconstructions being processed.

          - ``ksz_bv``: Per-object KSZ velocity bias $b_v^j$, defined by the equation 
            $\tilde T(\theta_j) = b_v^j v_r(x_j) + (\mbox{noise})$. This is only used to assign a normalization
            to power spectra $P_{gv}$ and $P_{vv}$. Can be specified as either:

              1. an array of length ``randcat.size``, to represent an arbitrary per-object $b_v$.
              2. a callable function $z \rightarrow f(z)$, if $b_v$ only depends on $z$.
              3. a scalar, if $b_v$ is the same for all galaxies.
              4. a length-``nksz`` list (or iterable) of any of 1-3, if $b_v$ is not the same for all
                 velocity reconstructions being processed. (This will usually be the case, see next paragraph.)

            The fiducial bias $b_v$ will depend on a choice of fiducial $P_{ge}(k)$, as well as the filter
            applied to $T_{CMB}$. (Since different velocity reconstructions use different CMB filters, $b_v$
            will usually depend on the velocity reconstruction.)

            One reasonable way of initializing $b_v$ is to use the following approximate expression
            from the overleaf:
            $$\begin{align}
            b_j^v &\approx B_v(z_j) \, W_{\rm CMB}(\theta_j) \\
            B_v(\chi) &\equiv \frac{K(\chi)}{\chi^2} \int \frac{d^2L}{(2\pi)^2} \, b_L F_L \, P_{ge}^{\rm true}(k,\chi)_{k=L/\chi}
            \end{align}$$

          - ``ksz_tcmb_realization``: 2-d array of shape ``(nksz, randcat.size)``. This is the filtered CMB $\tilde T$, 
            evaluated at locations of the randoms. This is used when making surrogate sims, to generate bootstrap realizations
            of the reconstruction noise.

            Note that each velocity reconstruction can use a different CMB filter. This is why $\tilde T$ is a 2-d
            array, indexed by ``0 <= i < nksz``, in addition to an index ``0 <= j < randcat.size``.

          - ``ztrue_col`` and ``zobs_col`` (strings): In general, the KszPSE allows the galaxy catalog to be
            photometric, with distinct redshifts $z_{\rm true}, z_{\rm obs}$. A spectroscopic catalog is the
            special case $z_{\rm true} = z_{\rm obs}$.

            The ``ztrue_col`` and ``zobs_col`` args are the names of the $z_{\rm true}, z_{\rm obs}$ columns
            in ``randcat``. The defaults (``ztrue_col = zobs_col = 'z'``) correspond to a spectroscopic catalog,
            with a column ``'z'`` containing redshifts. If the catalog is photometric (or if the redshift column
            is not ``z``), then you'll want to use something different.

          - ``deltac`` (float, default=1.68): This parameter is only used in surrogate sims with $f_{NL} \ne 0$, 
            to compute non-Gaussian bias from Gaussian bias, using $b_{ng} = delta_c (b_g - 1)$.

          - ``kernel`` (string, default=``cubic``): Interpolation kernel passed to ``kszx.interpolate_points()``.
            when simulating the surrogate field. Currently ``cic`` and ``cubic`` are implemented
            (will define more options later).

          - ``surr_ic`` (boolean): If True (the default), then the surrogates will satisfy a global
            "integral constraint" (gal-rand) = 0.

          - ``use_dc`` (boolean): if False (the default), then the k=0 mode will not be used,
            even if the lowest bin includes k=0.
        """
        
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
        self.surr_ic = surr_ic
        self.use_dc = use_dc

        ztrue = self._get_zcol(randcat, 'ztrue_col', ztrue_col)
        zobs = self._get_zcol(randcat, 'zobs_col', zobs_col)

        # Small optimization here: if (ztrue is zobs), then don't call ra_dec_to_xyz() twice.
        self.rcat_xyz_true = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=ztrue))
        self.rcat_xyz_obs = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=zobs)) if (ztrue is not zobs) else self.rcat_xyz_true
        
        fname = 'KszPSE.__init__()'
        nrand = self.nrand

        # 1-d arrays of length self.nrand (self.rweights can be None)
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

        # Check that we have enough randoms to make surrogate fields.
        self.bD_max = np.max(self.surr_bg * self.D)
        self.ngal_min = surr_ngal_mean - 3*surr_ngal_rms  # clamped to 3 sigma
        self.ngal_max = surr_ngal_mean + 3*surr_ngal_rms  # clamped to 3 sigma
        self.nrand_min = int(self.bD_max**2 * self.sigma2 * (self.ngal_max + 10))
        assert self.ngal_min >= 10
        
        if self.nrand < self.nrand_min:
            raise RuntimeError(f'KszPSE: not enough randoms to make surrogate fields! This can be fixed by using'
                               + f' a larger random catalog (nrand={self.nrand}, nrand_min={self.nrand_min}).')

        # pse_rweights = Length (nksz+1) list of (1-d array or None)
        pse_rweights = [ self.rweights ]   # can be None

        for (w,bv) in zip(self.ksz_rweights, self.ksz_bv):
            pse_rweights += [ (w*bv) if (w is not None) else bv ]

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

        # Parse the 'gweights', 'ksz_gweights', 'ksz_bv', and 'ksz_tcmb' args.
        # (Same parsing logic as 'rweights', 'ksz_rweights', 'ksz_bv', and 'ksz_tcmb_realization' in __init__().)

        fname = 'KszPSE.eval_pk()'
        gweights = self._parse_gal_arg(gweights, fname, 'gweights', gcat.size, z, non_negative=True, allow_none=True)
        ksz_bv = self._parse_ksz_arg(ksz_bv, fname, 'ksz_bv', gcat.size, z, allow_none=False)
        ksz_gweights = self._parse_ksz_arg(ksz_gweights, fname, 'ksz_gweights', gcat.size, z, allow_none=True)
        ksz_tcmb = self._parse_tcmb_arg(ksz_tcmb, fname, 'ksz_tcmb', gcat.size)
        assert len(ksz_bv) == len(ksz_gweights) == len(ksz_tcmb) == self.nksz

        # Initialize fmaps.

        gweights = gweights if (gweights is not None) else 1.0
        fmaps = [ self.catalog_gridder.grid_density_field(gcat, gweights, 0, zcol_name=zobs_col) ]

        for i in range(self.nksz):
            w, bv, t = ksz_gweights[i], ksz_bv[i], ksz_tcmb[i]
            coeffs = (w*t) if (w is not None) else t
            wsum = np.dot(w,bv) if (w is not None) else np.sum(bv)
            fmaps += [ self.catalog_gridder.grid_sampled_field(gcat, coeffs, wsum, i+1, spin=1, zcol_name=zobs_col) ]

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
            ``surr_ngal_mean`` and ``surr_ngal_rms`` in the :class:`~kszx.KszPSE` constructor
            docstring for more info.)

          - ``self.Sg_coeffs`` (1-d array of length ``self.nrand``): Coefficients $S_g^j$ above.

          - ``self.Sv_noise`` (shape-``(self.nksz, self.nrand)`` array): Coefficients $S_v^j$ above,
            contribution from reconstruction noise only.
        
          - ``self.Sv_signal`` (shape-``(self.nksz, self.nrand)`` array): Coefficients $S_v^j$ above,
            contribution from velocity field only (i.e. no reconstruction noise).

          - ``self.dSg_dfNL`` (1-d array of length ``self.nrand``): Derivative $dS_g^j/df_{NL}$.

        Note: Surrogates are factored into two functions (``simulate_surrogate()``and
        ``eval_pk_surrogate()`` so that the caller can put filtering logic in between, by
        operating directly on the ``self.Sg_coeffs``, ``self.Sv_noise``, ``Sv_signal``,
        and ``self.dSg_dfNL`` arrays. An example of filtering logic is subtracting the
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

        # delta_g(k,z) = (bg + 2 fNL deltac (bg-1) / alpha(k,z)) * delta_m(k,z)
        #              = (bg D(z) + 2 fNL deltac (bg-1) / alpha0(k)) * delta0(k)

        Sg_prefactor = ((ngal/nrand) * self.rweights) if (self.rweights is not None) else (ngal/nrand)
        
        # deltaG = (bg * delta_m) = (bg * D * delta0), evaluated on randcat.
        bD = self.surr_bg * self.D
        delta_G = bD * core.interpolate_points(self.box, delta0, self.rcat_xyz_true, self.kernel, fft=True)
        self.Sg_coeffs = Sg_prefactor * delta_G

        eta_rms = np.sqrt((nrand/ngal) - (bD*bD) * self.sigma2)
        eta = np.random.normal(scale = eta_rms)
        self.Sg_coeffs += Sg_prefactor * eta

        # Add term to deltag:
        #     2 fNL deltac (bg-1) / alpha(k,z) * delta_m(k,z)
        #   = 2 fNL deltac (bg-1) / alpha_z0(k) * delta0(k)    [ factor D(z) cancels ]
        phi0 = core.multiply_kfunc(self.box, delta0, lambda k: 1.0/self.cosmo.alpha_z0(k=k), dc=0)
        phi0 = core.interpolate_points(self.box, phi0, self.rcat_xyz_true, self.kernel, fft=True)
        self.dSg_dfnl = Sg_prefactor * (2 * self.deltac) * (self.surr_bg-1) * phi0

        if self.surr_ic:
            self.Sg_coeffs -= np.mean(self.Sg_coeffs)
            self.dSg_dfnl -= np.mean(self.dSg_dfnl)
        
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
            fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.Sv_noise[i,:], Sv_wsum, i+1, spin=1, zcol_name=self.zobs_col) ]
            fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.Sv_signal[i,:], Sv_wsum, i+1, spin=1, zcol_name=self.zobs_col) ]

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
            raise RuntimeError(f"KszPSE.reduce_pk(): 'pk' argument has invalid shape {pk.shape}")

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
            raise RuntimeError(f"KszPSE: expected argument {zcol_argname}={zcol_name} to be a string")

        if zcol_name not in catalog.col_names:
            raise RuntimeError(f"KszPSE: catalog does not contain column '{zcol_name}' (specified as argument '{zcol_argname}')")

        z = getattr(catalog, zcol_name)
        zmin = np.min(z)
        
        if zmin <= 0:
            raise RuntimeError(f"KszPSE: expected all redshifts to be positive (got min(catalog.{zcol_name}) = {zmin})")

        return z

    
    def _parse_gal_arg(self, f, funcname, argname, ngal, z, allow_none=False, non_negative=False):
        """Used to parse constructor args 'surr_bg' and 'rweights'.

        The 'f' argument is either:
           1. a callable function z -> f(z)
           2. an array of length ngal
           3. a scalar
           4. None

        If 'z' is None, then a function (option #1 above) is not allowed.
        If 'allow_none' is False, then None (option #4 above) is not allowed.

        Returns an array of length ngal (or None, if allow_none=True).
        """

        if f is None:
            if allow_none:
                return None
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
           4. None
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
