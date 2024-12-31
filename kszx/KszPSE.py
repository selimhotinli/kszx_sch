import numpy as np

from .Box import Box
from .Catalog import Catalog
from .Cosmology import Cosmology
from .CatalogGridder import CatalogGridder

from . import utils
from . import core


class KszPSE:
    def __init__(self, box, cosmo, randcat, kbin_edges, surr_ngal_mean, surr_ngal_rms, surr_bg, surr_bv=1.0, rweights=None, nksz=0, ksz_rweights=None, ksz_bv=None, ksz_tcmb_realization=None, photometric=None, deltac=1.68, kernel='cubic', use_dc=False):
        r"""KszPSE: High-level pipeline class (built on top of :class:`~kszx.CatalogGridder`) for $P_{gg}$, $P_{gv}$, and $P_{vv}$.

        Features:

          - Processes multiple velocity reconstructions efficiently in parallel -- for example,
            separate velocity reconstructions for 90 and 150 GHz. This feature can also be used
            to efficiently explore multiple possibilities for the foreground mask, CMB filtering,
            etc.

          - Runs "surrogate" sims (see overleaf) to characterize the survey window function 
            and assign error bars to power spectra.

          - Surrogate sims allow $f_{NL} \ne 0$, to characterize effect of $f_{NL}$ on windowed power 
            spectra, down to arbitrarily low $k$.

          - Velocity reconstruction noise is included in surrogate sims via a bootstrap procedure,
            using the observed CMB realization. This automatically incorporates noise inhomogeneity
            and "striping", and captures correlations e.g. between 90 and 150 GHz.

          - The galaxy catalog can be spectroscopic or photometric. Surrogate sims will capture
            the effect of photo-z errors.

          - The windowed power spectra $P_{gg}$, $P_{gv}$, $P_{vv}$ use a normalization which
            should be approximately correct. Thie normalization is an ansatz which is imperfect,
            especially on large scales, so surrogate sims should still be used to compare power
            spetra to models. (However, it's convenient to have a normalization which is not
            "off" by orders of magnitude!) Eventually, we'll implement precise window function
            deconvolution.

        Usage:

          - Construct a KszPSE object.

          - To evaluate power spectra ($P_{gg}$, $P_{gv}$, or $P_{vv}$) on **data or a mock**, 
            call :meth:`KszPSE.eval_pk()` with a galaxy catalog (``gcat``), and filtered CMB 
            temperatures at the galaxy locations (``ksz_tmcb``).

          - To evaluate power spectra on a **surrogate simulation**, call 
            :meth:`KszPSE.simulate_surrogate()` to make the simulation, then call
            :meth:`KszPSE.eval_pk_surrogate()` to estimate power spectra.
        
        For examples of KszPSE in action, see:

          - **Add links to jupyter notebook(s) here**

        For a formal specification of what KszPSE computes, see ":ref:`ksz_pse_details`" 
        in the sphinx docs.

        **Explain per-object bias $b_i^v$, with equations.**

        **Explain bootstrap method for the noise (``ksz_tcmb_realization``).**

        **We assume only one galaxy field (and only one random field) for simplicity.
        (This could easily be generalized if needed.)**

        **Discussion of surrogate $N_{gal}$ Gaussian distribution.**

        **Make the point somewhere that surrogates are factored into two functions on purpose.**

        Constructor args:

          - ``box`` (:class:`~kszx.Box`): defines pixel size, bounding box size, and location of observer.

          - ``cosmo`` (:class:`~kszx.Cosmology`).

          - ``randcat`` (:class:`~kszx.Catalog`): random catalog, defines survey footprint (including
            redshift distribution). The randcat must contain columns ``ra_deg`` and ``dec_deg``, and
            either:

              1. A redshift column ``z``, but not ``ztrue`` or ``zobs`` (spectroscopic catalog).

              2. Redshift columns ``ztrue`` and ``zobs``, but not ``z`` (photometric catalog).
                 In this case, the random catalog must accurately simulate the joint 
                 $(z_{true}, z_{obs})$ distribution of the galaxies.
        
            (Reminder: you can use ``Catalog.add_column()`` and ``Catalog.remove_column()``
            to manipulate Catalog columns.)

          - ``kbin_edges`` (array): 1-d array of length (nkbins+1) defining $k$-bin endpoints for
            power spectrum estimation. The i-th bin covers k-range ``kbin_delim[i] <= i < kbin_delim[i+1]``.

          - ``surr_ngal_mean`` (float): Mean of Gaussian $N_{gal}$ distribution for 
            surrogate sims (see above).

          - ``surr_ngal_rms`` (float): RMS width of Gaussian $N_{gal}$ distribution for 
            surrogate sims (see above).

          - ``surr_bg``: Galaxy bias $b_g$ used in surrogate sims. Can be specified as either:

               1. an array of length ``randcat.size``, to represent an arbitrary per-object $b_g$.
               2. a callable function $z \rightarrow f(z)$, if $b_g$ only depends on $z$.
               3. a scalar, if $b_g$ is the same for all galaxies.

          - ``surr_bv`` (float, optional): KSZ velocity reconstruction bias in surrogate sims.
            This is multiplied by the bias specified in ``ksz_bv``, which represents bias computed
            from a fiducial $P_{ge}(k)$. The ``surr_bv`` argument should be specified if you want
            the surrogate sims to have a $b_v$ which differs from its fiducial value.

          - ``rweights`` (optional): Galaxy weighting $W_i^L$ used for the large-scale galaxy field $\delta_g(x)$.
            Can be specified as either:

               1. an array of length ``randcat.size``, to represent an arbitrary per-object $W_i^L$.
               2. a callable function $z \rightarrow f(z)$, if $W_i^L$ only depends on $z$.
               3. a scalar, if $W_i^L$ is the same for all galaxies.
               4. None (equivalent to ``rweights=1.0``).

            (Note that there is also a galaxy weighting $W_i^S$ used in the kSZ velocity reconstruction $\hat v_r(x)$.
            This weighting is independent, and is specified as the ``ksz_rweights`` constructor arg, see beow.)

          - ``nksz`` (integer): Number of kSZ velocity reconstruction fields $\hat v_r(x)$.
            For example, to separate velocity reconstructions from 90 and 150 GHz, use ``nksz=2``.
            See above for more discussion. If ``nksz`` is zero or unspecified, then the KszPSE will
            estimate the power spectrum of a galaxy field $\delta_g$, but no $\hat v_r$ fields.

          - ``ksz_rweights`` (optional): Galaxy weighting $W_i^S$ used for the kSZ velocity reconstruction $\hat v_r(x)$.
            Can be specified as either:

              1. an array of length ``randcat.size``, to represent an arbitrary per-object $W_i^S$.
              2. a callable function $z \rightarrow f(z)$, if $W_i^S$ only depends on $z$.
              3. a scalar, if $W_i^S$ is the same for all objects.
              4. None (equivalent to ``ksz_rweights=1``).
              5. a length-``nksz`` list (or iterable) of any of 1-4, if $W_i^S$ is not the same for all
                 velocity reconstructions being processed.

          - ``ksz_bv``: Fiducial KSZ velocity bias $b_v$. See above for more discsusion, and some key equations
            showing how to compute $b_v$. Can be specified as either:

              1. an array of length ``randcat.size``, to represent an arbitrary per-object $b_v$.
              2. a callable function $z \rightarrow f(z)$, if $b_v$ only depends on $z$.
              3. a scalar, if $b_v$ is the same for all galaxies.
              4. a length-``nksz`` list (or iterable) of any of 1-3, if $b_v$ is not the samee for all
                 velocity reconstructions bing processed. (This will usually be the case.)

            The fiducial bias $b_v$ will depend on a choice of fiducual $P_{ge}(k)$, as well as the filter
            applied to $T_{CMB}$. (Note that each velocity reconstruction can use a different CMB filter, so
            $b_v$ can depend on an index ``0 <= i < nksz``, in addition to an index ``0 <= j < randcat.size``)
        
          - ``ksz_tcmb_realization``: 2-d array of shape ``(nksz, randcat.size)``. This is the filtered CMB $\tilde T$, 
            evaluated at locations of the randoms. (Note that each velocity reconstruction can use a different CMB filter,
            so $\tilde T$ can depend on an index ``0 <= i < nksz``, in addition to an index ``0 <= j < randcat.size``)

          - ``photometric`` (boolean, optional): Indicates whether galaxy catalog is photometric
            or spectroscopic. The default (``photometric=None``) is to autodetect whether
            the randcat is photometric, based on which redshift columns are defined (see
            ``randcat`` above). However, if the randcat is photometric and you specify
            ``photometric=False``, then the SurrogateFactory will ignore the 'zobs'
            column, and define a spectroscopic catalog using the 'ztrue' column.

          - ``deltac`` (float, default=1.68): This parameter is only used in surrogate sims with $f_{NL} \ne 0$, 
            to compute non-Gaussian bias from Gaussian bias, using $b_{ng} = delta_c (b_g - 1)$.

          - ``kernel`` (string, default=``cubic``): Interpolation kernel passed to ``kszx.interpolate_points()``.
            when simulating the surrogate field. Currently ``cic`` and ``cubic`` are implemented
            (will define more options later).

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
        self.kbin_edges = core._check_kbin_delim(box, kbin_edges, use_dc)
        self.nkbins = len(kbin_edges)-1
        self.nksz = nksz
        self.surr_ngal_mean = surr_ngal_mean
        self.surr_ngal_rms = surr_ngal_rms
        self.nrand = randcat.size
        self.deltac = deltac
        self.kernel = kernel
        self.use_dc = use_dc
        
        ztrue, zobs, photometric = self._init_redshifts(randcat, photometric)
        assert np.min(ztrue) >= 0
        assert np.min(zobs) >= 0
        
        self.photometric = photometric
        self.ztrue = ztrue
        self.zobs = zobs

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
        self.ksz_tcmb_realization = self._parse_ksz_arg(ksz_tcmb_realization, fname, 'ksz_tcmb_realization', nrand, z=None, allow_none=False)
        assert len(self.ksz_bv) ==  len(self.ksz_rweights) == len(self.ksz_tcmb_realization) == self.nksz

        # 1-d array of length nksz.
        self.surr_bv = self._parse_surr_bv(surr_bv)
        assert self.surr_bv.shape == (self.nksz,)
        
        # 1-d arrays of length self.nrand
        self.D = cosmo.D(z=ztrue, z0norm=True)
        self.faH = cosmo.frsd(z=ztrue) * cosmo.H(z=ztrue) / (1+ztrue)
        self.sigma2 = self._integrate_kgrid(self.box, cosmo.Plin_z0(self.box.get_k()))

        # Check that we have enough randoms to make surrogate fields.
        bD_max = np.max(self.surr_bg * self.D)
        ngal_max = surr_ngal_mean + 3*surr_ngal_rms
        self.nrand_min = int(ngal_max * bD_max + 10)
        
        if self.nrand < self.nrand_min:
            raise RuntimeError(f'KszPSE: not enough randoms to make surrogate fields! This can be fixed by using'
                               + f' a larger random catalog (nrand={self.nrand}, nrand_min={self.nrand_min}),'
                               + f' decreasing {surr_ngal_mean=}, decreasing {surr_ngal_rms=}, or decreasing'
                               + f' galaxy bias (current max bD = {bD_max})')

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
            zcol_name = ('zobs' if photometric else 'z'),
            save_rmaps = [True] + [False]*nksz,
            kernel = self.kernel
        )

    
    def eval_pk(self, gcat, gweights=None, ksz_gweights=None, ksz_bv=None, ksz_tcmb=None, zcol_name='z'):
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
            passed to the :class:`~kszx.KszPSE` constructor. For example, if one is FKP-weighted, then the
            other should also be FKP-weighted. (Warning: even a small mismatch between the weighting of
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
    
          - ``ksz_bv``: Fiducial KSZ velocity bias $b_v$. See above for more discsusion, and some key equations
            showing how to compute $b_v$. Can be specified as either:

              1. an array of length ``gcat.size``, to represent an arbitrary per-object $b_v$.
              2. a callable function $z \rightarrow f(z)$, if $b_v$ only depends on $z$.
              3. a scalar, if $b_v$ is the same for all galaxies.
              4. a length-``nksz`` list (or iterable) of any of 1-3, if $b_v$ is not the samee for all
                 velocity reconstructions bing processed. (This will usually be the case.)

            The fiducial bias $b_v$ will depend on a choice of fiducual $P_{ge}(k)$, as well as the filter
            applied to $T_{CMB}$. (Note that each velocity reconstruction can use a different CMB filter, so
            $b_v$ can depend on an index ``0 <= i < nksz``, in addition to an index ``0 <= j < randcat.size``)

            The ``ksz_bv`` argument passed to ``eval_pk()`` should reflect as closely as possible the 
            ``ksz_bv`` argument passed to the :class:`~kszx.KszPSE` constructor. (Same filtering, etc.)

          - ``ksz_tcmb``: 2-d array of shape ``(nksz, gcat.size)``.  This is the filtered CMB $\tilde T$, 
            evaluated at the location of the galaxies. (Note that each velocity reconstruction can use a different CMB filter,
            so $\tilde T$ can depend on an index ``0 <= i < nksz``, in addition to an index ``0 <= j < gcat.size``)

            The ``ksz_tcmb`` argument passed to ``eval_pk()`` should reflect as closely as possible the 
            ``ksz_tcmb_realization`` argument passed to the :class:`~kszx.KszPSE` constructor. (Same filtering, etc.)
        
          - ``zcol_name`` (string): The galaxy catalog ``gcat`` must contain columns ``'ra_deg'``,
            ``'dec_deg'``, and a redshift column which is ``'z'`` by default, but this can
            be overridden by specifying the ``zcol_name`` argument.

            For example, in a photometric catalog, you might have two columns ``'ztrue'``
            and ``'zobs'``. In this case, you probably want ``zcol_name = 'zobs'``, in
            order to grid the catalog at observed redshifts, not true (unobserved) redshifts.

        Return value:

          - Array ``pk`` with shape ``(nksz+1, nksz+1, nkbins)``. The first two indices are
            field indices, with ordering (galaxy field, vrec fields). Thus the ``pk`` array
            contains all auto and cross power spectra $P_{gg}$, $P_{gv_i}$, $P_{v_iv_j}$.
        """

        assert isinstance(gcat, Catalog)
        assert zcol_name in gcat.col_names

        z = getattr(gcat, zcol_name)
        gcat_xyz = gcat.get_xyz(self.cosmo, zcol_name)

        # Parse the 'gweights', 'ksz_gweights', 'ksz_bv', and 'ksz_tcmb' args.
        # (Same parsing logic as 'rweights', 'ksz_rweights', 'ksz_bv', and 'ksz_tcmb_realization' in __init__().)

        fname = 'KszPSE.eval_pk()'
        ksz_bv = self._parse_ksz_arg(ksz_bv, fname, 'ksz_bv', gcat.size, z, allow_none=False)
        ksz_gweights = self._parse_ksz_arg(ksz_gweights, fname, 'ksz_gweights', gcat.size, z, allow_none=True)
        ksz_tcmb = self._parse_ksz_arg(ksz_tcmb, fname, 'ksz_tcmb', gcat.size, z=None, allow_none=False)
        gweights = self._parse_gal_arg(gweights, fname, 'gweights', gcat.size, z, non_negative=True, allow_none=True)
        assert len(ksz_bv) == len(ksz_gweights) == len(ksz_tcmb) == self.nksz

        # Initialize fmaps.

        gweights = gweights if (gweights is not None) else 1.0
        fmaps = [ self.catalog_gridder.grid_density_field(gcat, gweights, 0, zcol_name=zcol_name) ]

        for i in range(self.nksz):
            w, bv, t = ksz_gweights[i], ksz_bv[i], ksz_tcmb[i]
            coeffs = (w*t) if (w is not None) else t
            wsum = np.dot(w,bv) if (w is not None) else np.sum(bv)
            fmaps += [ self.catalog_gridder.grid_sampled_field(gcat, coeffs, wsum, i+1, spin=1, zcol_name=zcol_name) ]

        # FIXME need some sort of CatalogPSE helper function here.
        pk = core.estimate_power_spectrum(self.box, fmaps, self.kbin_edges)
        pk *= np.reshape(self.catalog_gridder.ps_normalization, (self.nksz+1, self.nksz+1, 1))
        return pk
        

    def simulate_surrogate(self, enable_fnl=False):
        r"""Makes a random surrogate simulation, and stores the result in class members (return value is None).
        
        $$\begin{align}
        S_g(x) &= \sum_{j\in\rm rand} S_g^j \delta^3(x-x_j) \\
        S_v(x) &= \sum_{j\in\rm rand} S_v^j \delta^3(x-x_j)
        \end{align}$$

        Initializes the following class members:

          - ``self.surr_ngal`` (integer):

          - ``self.Sg_coeffs`` (1-d array of length ``self.nrand``): Coefficients $S_g^j$ above.

          - ``self.Sv_coeffs`` (shape-``(self.nksz, self.nrand)`` array): Coefficients $S_v^j$ above.
            (Note that each velocity reconstruction uses different bias and noise, so $S_v^j$
            depends on an index``0 <= i < nksz``, in addition to an index ``0 <= j < randcat.size``)

          - ``self.dSg_dfNL`` (1-d array of length ``self.nrand``): Derivative $dS_g^j/df_{NL}$,
            only computed if ``enable_fnl=True``.

        For a formal specification of what KszPSE computes, see ":ref:`ksz_pse_details`" 
        in the sphinx docs.
        """

        ngal = self.surr_ngal_mean + (self.surr_ngal_rms * np.clip(np.random.normal(), -3.0, 3.0))
        ngal = int(ngal+0.5)  # round
        self.surr_ngal = ngal
        
        nrand = self.nrand
        assert 0 < ngal <= nrand
                
        delta0 = core.simulate_gaussian_field(self.box, self.cosmo.Plin_z0)
        core.apply_kernel_compensation(self.box, delta0, self.kernel, -0.5)  # multiply by C(k)^(-1/2)

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
        self.dSg_dfnl = None

        if enable_fnl:
            # Add term to deltag:
            #     2 fNL deltac (bg-1) / alpha(k,z) * delta_m(k,z)
            #   = 2 fNL deltac (bg-1) / alpha_z0(k) * delta0(k)    [ factor D(z) cancels ]
            phi0 = core.multiply_kfunc(self.box, delta0, lambda k: 1.0/self.cosmo.alpha_z0(k=k), dc=0)
            phi0 = core.interpolate_points(self.box, phi0, self.rcat_xyz_true, self.kernel, fft=True)
            self.dSg_dfnl = Sg_prefactor * (2 * self.deltac) * (self.surr_bg-1) * phi0
            
        self.Sv_coeffs = np.zeros((self.nksz, self.nrand))
        
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
            self.Sv_coeffs[i,:] = ((ngal/nrand) * self.surr_bv[i]) * self.ksz_bv[i] * vr
            self.Sv_coeffs[i,:] += M * self.ksz_tcmb_realization[i]

            if self.ksz_rweights[i] is not None:
                self.Sv_coeffs[i,:] *= self.ksz_rweights[i]

    
    def eval_pk_surrogate(self, fnl=0.0):
        r"""Computes $P_{gg}$, $P_{gv}$, $P_{vv}$ from a surrogate sim (from the most recent call to :meth:`simulate_surrogate()`).

        If the ``fnl`` argument is a scalar, then the return value from ``eval_pk_surrogate()`` is an array ``pk`` with 
        shape ``(nksz+1, nksz+1, nkbins)``. The first two indices are field indices, with ordering (galaxy field, vrec 
        fields). Thus the ``pk`` array contains all auto and cross power spectra $P_{gg}$, $P_{gv_i}$, $P_{v_iv_j}$.

        If the ``fnl`` argument is a 1-d array of length ``nfnl``, then the return value from ``eval_pk_surrogate()`` 
        is an array ``pk`` with shape ``(nfnl, nksz+1, nksz+1, nkbins)``, containing power spectra for each specified
        value of $f_{NL}$. (A typical use case is ``fnl=[-100,0,100]``, to compute power spectra for a few $f_{NL}$
        values, in order to explore $f_{NL}$ derivatives.)
        """

        if not hasattr(self, 'Sg_coeffs'):
            raise RuntimeError("Before calling KszPSE.eval_pk_surrogate(), you must call KszPSE.simulate_surrogate()")
        
        fnl = utils.asarray(fnl, 'KszPSE.eval_pk_surrogate()', 'fnl', dtype=float)
        if (fnl.ndim > 1) or (fnl.size == 0):
            raise RuntimeError(f"Expected 'fnl' to be 0-d or 1-d array, got shape {fnl.shape}")

        fnl_is_nonzero = np.any(fnl != 0)
        if fnl_is_nonzero and (self.dSg_dfnl is None):
            raise RuntimeError("Before calling KszPSE.eval_pk_surrogate() with nonzero fnl,"
                               + " you must call KszPSE.simulate_surrogate() with enable_fnl=True")
        
        Nr = self.nrand
        Ng = self.surr_ngal
        zcol_name = ('zobs' if self.photometric else 'z')
        
        Sg_wsum = ((Ng/Nr) * np.sum(self.rweights)) if (self.rweights is not None) else Ng
        fmaps = [ self.catalog_gridder.grid_sampled_field(self.randcat, self.Sg_coeffs, Sg_wsum, 0, zcol_name=zcol_name) ]
            
        for i in range(self.nksz):
            w, bv = self.ksz_rweights[i], self.ksz_bv[i]
            Sv_wsum = (Ng/Nr) * (np.dot(w,bv) if (w is not None) else np.sum(bv))
            fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.Sv_coeffs[i,:], Sv_wsum, i+1, spin=1, zcol_name=zcol_name) ]
        
        if fnl_is_nonzero:
            fmaps += [ self.catalog_gridder.grid_sampled_field(self.randcat, self.dSg_dfnl, Sg_wsum, 0, zcol_name=zcol_name) ]

        nfnl = fnl.size
        fnl_1d = np.reshape(fnl, (nfnl,))
        
        # pk0.shape = (nksz+2, nksz+2, nkbins)    ordering (Sg,Sv,dSg_dfnl)
        pk0 = np.zeros((self.nksz+2, self.nksz+2, self.nkbins))
        pk0[:len(fmaps), :len(fmaps), :] = core.estimate_power_spectrum(self.box, fmaps, self.kbin_edges)

        pk_ret = np.zeros((nfnl, self.nksz+1, self.nksz+1, self.nkbins))

        for i,f in enumerate(fnl_1d):
            # pk1.shape = (nksz+1, nksz+2, nkbins)
            pk1 = pk0[:-1,:,:].copy()
            pk1[0,:,:] += f * pk0[-1,:,:]

            # pk2.shape = (nksz+1, nksz+1, nkbins)
            pk2 = pk1[:,:-1,:].copy()
            pk2[:,0,:] += f * pk1[:,-1,:]
            
            pk_ret[i,:,:,:] = pk2
            
        # Apply normalization
        pk_ret *= np.reshape(self.catalog_gridder.ps_normalization, (1, self.nksz+1, self.nksz+1, 1))

        dst_shape = fnl.shape + (self.nksz+1, self.nksz+1, self.nkbins)
        return np.reshape(pk_ret, dst_shape)

        
    ####################################################################################################

    
    def _init_redshifts(self, randcat, photometric):
        """Helper method called by constructor. Returns (rcat_ztrue, rcat_zobs, photometric). 

        The 'photometric' arg can be None (for "autodetect from randcat")."""

        zcols = set(['z','ztrue','zobs'])
        zcols = zcols.intersection(randcat.col_names)

        # Case 1: spectroscopic catalog.
        if zcols == set(['z']):
            if photometric is None:
                print("KszPSE: setting photometric=False (randcat contains 'z' but not ztrue/zobs)")
            elif photometric:
                raise RuntimeError("photometric=True was specified, but randcat only contains 'z' column")
            return randcat.z, randcat.z, False

        # Case 2: photometric catalog.
        if zcols == set(['ztrue','zobs']):
            if photometric is None:
                print("KszPSE: setting photometric=True (catalog contains ztrue/zobs but not 'z')")
            elif not photometric:
                print("KszPSE: randcat is photometric, but photometric=False was specified,"
                      " setting zobs=ztrue to mimic spectroscopic catalog")
                return randcat.ztrue, randcat.ztrue, False
            else:
                return randcat.ztrue, randcat.zobs, True

        raise RuntimeError("Expected one of two cases: (1) spectroscopic randcat containing 'z' but not ('ztrue','zobs'),"
                           + " or (2) photometric randcat containing ('ztrue','zobs') but not 'z'. Actual randcat contains"
                           + f" the following columns: {sorted(randcat.col_names)}")

    
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
            raise RuntimeError(f"{funcname}: couldn't parse constructor arg '{argname}' (got length-{flen} iterable non-array,"
                               + f" expected array, function, scalar, None, or iterable with length nksz={self.nksz})")

        # This case can be handled by multiple calls to self._parse_gal_arg().
        return [ self._parse_gal_arg(x, funcname, f'{argname}[{i}]', ngal, z, allow_none=allow_none, non_negative=non_negative) for i,x in enumerate(f) ]


    def _parse_surr_bv(self, surr_bv):
        if surr_bv is None:
            surr_bv = 1.0

        surr_bv = utils.asarray(surr_bv, 'KszPSE', 'surr_bv', dtype=float)

        if surr_bv.shape == (self.nksz,):
            return surr_bv
        if surr_bv.ndim == 0:
            return np.full(self.nksz, float(surr_bv))

        raise RuntimeError("KszPSE: expected 'surr_bv' constructor arg to be either a scalar,"
                           + f" or a 1-d array of length nksz={self.nksz}, got shape {surr_bv.shape}")

    
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
