import numpy as np

from .Box import Box
from .Catalog import Catalog
from .Cosmology import Cosmology

from . import utils
from . import core


class SurrogateFactory:
    def __init__(self, box, cosmo, randcat, ngal_mean, ngal_rms, bg, rweights=None, fnl=0, ksz=False, bvr=None, vr_noise_realization=None, vr_rms=None, photometric=None, deltac=1.68, kernel='cubic'):
        """SurrogateFactory: high-level class which simulates "surrogate" fields for a galaxy survey.

           <eta^2> = (N_surr/N_gal) - < delta_G^2 >
        
        Constructor args:

          - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer,
            relative to random catalog. See :class:`~kszx.Box` for more info.

          - ``cosmo`` (kszx.Cosmology). This is needed because the SurrogateFactory simulates
            the density field, which needs a linear power spectrum, growth function, etc.

          - ``randcat`` (ksz.Catalog): random catalog that defines survey footprint.
            Must contain columns 'ra_deg' and 'dec_deg', and one of two possibilities
            for redshifts:
        
              1. spectroscopic: randcat contains 'z' column, but not 'ztrue' or 'zobs'.
              2. photometric: randcat contains 'ztrue' and 'zobs' columns, but not 'z'.

            Note that in the photometric case, your random catalog must contain 'ztrue' and
            'zobs' columns which accurately simulate the joint $(z_{\rm true}, z_{\rm obs})$
            distribution of the galaxies.
        
            (Reminder: you can use ``Catalog.add_column()`` and ``Catalog.remove_column()``
            to manipulate Catalog columns.)

          - ``ngal_mean``: Mean number of galaxies in the galaxy survey being simulated.
        
          - ``ngal_rms``: RMS scatter in number of survey galaxies. (FIXME revisit and explain in more detail.)

          - ``bg``: Galaxy bias, specified as one of the following:
        
               1. a function z -> b(z)
               2. a 1-d array of length randcat.size
               3. a scalar

          - ``rweights`` (optional): Per-object weights applied to the random catalog.
            For example, an FKP weighting. Can be specified in any of the three
            ways as the ``bg`` argumment (see above).

            You should use an ``rweights`` which is as consistent as possible with
            the weighting that you use to analyze the galaxy data. For example, if
            you plan to analyze galaxy data with FKP weighting, you should also
            use FKP weighting in the ``SurrogateFactory``.

          - ``fnl`` (float): The primordial NG parameter $f_{NL}$ (zero by default).
            Currently implemented as a pure scale-dependent galaxy bias, with no effect
            on the velocity field or the matter bispectrum.
        
          - ``ksz`` (boolean): If True, then a velocity reconstruction surrogate field
            will be simulated (in addition to the galaxy surrogate field). (For details,
            see ``SurrogateFactory.simulate()``.)

          - ``bvr``: Bias for the velocity reconstruction. Should be specified iff ``ksz=True``.
            Can be specified in any of the three ways as the ``bg`` argumment (see above).

          - ``vr_noise_realization``: Velocity reconstruction noise *realization* (not RMS).
            Must be specified as a 1-d array of length ``randcat.size``. Note that
            reconstruction noise can either be specified as a "realization" or an "RMS",
            see a few lines below for discussion.
        
          - ``vr_rms``: Per-object velocity reconstruction noise *RMS* (not realization).
            Can be specified in any of the three ways as the ``bg`` argument (see above).

            Note that velocity reconstruction noise can be specified in two ways: either
            as a noise *realization* (via the ``vr_noise_realization`` argument), or a
            noise *rms* (via the ``vr_rms`` argument). If ``vr_rms`` is specified, then
            an independent Gaussian noise realization will be simulated in each call to
            ``SurrogateFactory.simulate()``.

            If ``vr_noise_realization`` is specified, then the same noise realization
            will be recycled in each call to ``SurrogateFactory.simulate()``. This is
            intended for a case where we take the noise realization directly from a
            CMB map, without attempting to model the noise covariance.
        
          - ``photometric`` (boolean): Indicates whether galaxy catalog is photometric
            or spectroscopic. The default (``photometric=None``) is to autodetect whether
            the randcat is photometric, based on which redshift columns are defined (see
            ``randcat`` above). However, if the randcat is photometric and you specify
            ``photometric=False``, then the SurrogateFactory will ignore the 'zobs'
            column, and define a spectroscopic catalog using the 'ztrue' column.

          - ``deltac`` (float): This parameter is only used if (fnl != 0), to compute
            non-Gaussian bias from Gaussian bias, using $b_{ng} = \delta_c (b_g - 1)$.

          - ``kernel`` (string): Interpolation kernel passed to ``kszx.interpolate_points()``.
            when simulating the surrogate field. Currently ``cic`` and ``cubic`` are implemented
            (will define more options later).
        """

        assert isinstance(box, Box)
        assert isinstance(cosmo, Cosmology)
        assert isinstance(randcat, Catalog)
        assert kernel in [ 'cic', 'cubic' ]
        
        assert randcat.size > 0
        assert 'ra_deg' in randcat.col_names
        assert 'dec_deg' in randcat.col_names
        
        assert ngal_mean > 0
        assert ngal_rms > 5*ngal_mean
        assert ngal_mean + 3*ngal_rms <= randcat.size
        
        self.box = box
        self.cosmo = cosmo
        self.ngal_mean = ngal_mean
        self.ngal_rms = ngal_rms
        self.nrand = randcat.size
        self.fnl = fnl
        self.ksz = ksz
        self.deltac = deltac
        self.kernel = kernel
        # Note: we don't save reference to 'randcat', but we do save references to some of its columns.
        
        surr_ztrue, surr_zobs, photometric = self._init_redshifts(randcat, photometric)
        assert np.min(surr_ztrue) >= 0
        assert np.min(surr_zobs) >= 0
        
        self.photometric = photometric
        self.surr_ra_deg = randcat.ra_deg
        self.surr_dec_deg = randcat.dec_deg
        self.surr_ztrue = surr_ztrue
        self.surr_zobs = surr_zobs

        # These initializations must be done after calling _init_redshifts()
        self.bg = bg = self._eval_zfunc(bg, surr_ztrue, 'bg')
        self.bvr = bvr = self._eval_zfunc(bg, surr_ztrue, 'bvr', allow_none=True)
        self.vr_rms = vr_rms = self._eval_zfunc(bg, surr_ztrue, 'vr_rms', allow_none=True, non_negative=True)
        self.vr_noise_realization = vr_noise_realization = self._eval_zarr(vr_noise_realization, 'vr_noise_realization')
        self.rweights = rweights = self._eval_zfunc(rweights, surr_zobs, 'rweights', allow_none=True, non_negative=True)

        # Checks: if ksz=True, then bvr is not None, and precisely one of {vr_rms, vr_noise_realization} is None.
        self._check_ksz_args()

        self.surr_D = cosmo.D(z=surr_ztrue, z0norm=True)
        self.surr_faH = cosmo.f(z=surr_ztrue) * cosmo.H(z=surr_true) / (1+surr_ztrue)
        self.surr_xyz_true = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=surr_ztrue))
        self.surr_xyz_obs = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=surr_zobs))
        self.sigma2 = self._integrate_kgrid(self.box, cosmo.Plin_z0(self.box.get_k()))

        # Check: the following quantity is always nonnegative
        #   <eta^2> = (N_rand/N_gal) - < delta_G^2 >

        bD_max = np.max(bg * self.surr_D)
        ngal_max = ngal_mean + 3*ngal_rms
        self.nrand_min = int(ngal_max * + 10)
        
        if self.nrand < nrand_min:
            raise RuntimeError(f'SurrogateFactory: not enough randoms! This can be fixed by using a larger'
                               + f' random catalog (nrand={self.nrand}, nrand_min={self.nrand_min}),'
                               + f' decreasing {ngal_mean=}, decreasing {ngal_rms=}, or decreasing'
                               + f' galaxy bias (current max bD = {bD_max})')


    def simulate(self):
        """
        Initializes::

         self.surr_gweights  # needed for CatalogPSE ('weights' argument for surrogate galaxy field)
         self.surr_gvalues   # needed for CatalogPSE ('values' argument for surrogate galaxy field)
         self.surr_ngal      # not needed for CatalogPSE

        If ksz=True, additionally initializes::
        
         self.surr_bvr       # needed for CatalogPSE ('weights' argument for surrogate vrec field)
         self.surr_vr        # needed for CatalogPSE ('values' argument for surrogate vrec field)
        """

        t = np.clip(np.random.normal(), -3.0, 3.0)
        ngal = self.ngal_mean + t * self.ngal_rms
        nsurr = self.nrand
        
        delta0 = simulate_gaussian_field(self.box, self.cosmo.Plin_z0)
        delta0 /= np.sqrt(compensation_kernel(self.box, self.kernel))

        # delta_g(k,z) = (bg + 2 fNL deltac (bg-1) / alpha(k,z)) * delta_m(k,z)
        #              = (bg D(z) + 2 fNL deltac (bg-1) / alpha0(k)) * delta0(k)
        
        bD = self.bg * self.surr_D
        deltag = bD * core.interpolate_points(self.box, delta0, self.rcat_xyz_true, self.kernel, fft=True)

        if self.fnl != 0:
            phi0 = multiply_kfunc(self.box, delta0, lambda k: 1.0/self.cosmo.alpha_z0(k=k), dc=0)
            phi0 = core.interpolate_points(self.box, phi0, self.surr_xyz_true, self.kernel, fft=True)
            deltag += 2 * self.fnl * self.deltac * (self.bg-1) * phi0

        # Add noise to delta_g
        #  <eta^2> = (nsurr/ngal) - <deltag^2>
        #          = (nsurr/ngal) - bD^2 self.sigma2

        noise_var = (nsurr/ngal) - self.sigma2 * (bD*bD)
        deltag += np.random.normal(scale = np.sqrt(noise_var))
        
        self.surr_gweights = () * self._xx
        self.surr_gvalues = self.surr_gweights * deltag
        self.surr_ngal = ngal

        if self.ksz:
            vr = multiply_kfunc(self.box, delta0, lambda k: 1.0/k, dc=0)   # note wrong normalization (no faHD or faH0)
            vr = core.interpolate_points(self.box, vr, self.surr_xyz_true, self.kernel, fft=True, spin=1)
            vr *= (self.bvr * self.surr_faH * self.surr_D)                 # correct normalization
            nvr = self.vr_noise_realization if (self.vr_noise_realization is not None) else np.random.normal(scale=self.vr_rms)
            
            self.surr_bvr = (ngal/nsurr) * self.bvr
            self.surr_vr = (ngal/nsurr)*vr + np.sqrt(ngal/nsurr)*nvr
            

    ####################################################################################################
    

    def _init_redshifts(self, randcat, photometric):
        """Helper method called by constructor. Returns (rcat_ztrue, rcat_zobs, photometric). 

        The 'photometric' arg can be None (for "autodetect from randcat")."""

        zcols = set(['z','ztrue','zobs'])
        zcols = zcols.intersection(randcat.col_names)

        # Case 1: spectroscopic catalog.
        if zcols == set(['z']):
            if photometric is None:
                print("SurrogateFactory: setting photometric=False (randcat contains 'z' but not ztrue/zobs)")
            elif photometric:
                raise RuntimeError("photometric=True was specified, but randcat only contains 'z' column")
            return randcat.z, randcat.z, False

        # Case 2: photometric catalog.
        if zcols == set(['ztrue','zobs']):
            if photometric is None:
                print("SurrogateFactory: setting photometric=True (catalog contains ztrue/zobs but not 'z')")
            elif not photometric:
                print("SurrogateFactory: randcat is photometric, but photometric=False was specified,"
                      " setting zobs=ztrue to mimic spectroscopic catalog")
                return randcat.ztrue, randcat.ztrue, False
            else:
                return randcat.ztrue, randcat.zobs, True

        raise RuntimeError("Expected one of two cases: (1) spectroscopic randcat containing 'z' but not ('ztrue','zobs'),"
                           + " or (2) photometric randcat containing ('ztrue','zobs') but not 'z'. Actual randcat contains"
                           + f" the following columns: {sorted(randcat.col_names)}")

        
    def _eval_zfunc(self, f, z, name, allow_none=False, non_negative=False):
        """Helper method for contructor. Used to parse the 'bg', 'bvr', and 'vr_rms' constructor args.

        Returns an array of length self.nrand (or None, if allow_none=True).
        The 'f' argument is either a callable function z -> f(z), an array of length self.nrand, or a scalar.
        """

        if f is None:
            if allow_none:
                return None
            raise RuntimeError(f"SurrogateFactory: '{f}' constructor argument must be specified")
        
        if callable(f):
            fz = f(z)
            fz_name = f'return value from {name}()'
            fz = utils.asarray(fz, 'SurrogateFactory', fz_name, dtype=float)
            if fz.shape != (self.nrand,):
                raise RuntimeError(f"{fz_name} has shape {fz.shape}, expected shape {(self.nrand,)}")
        
        else:
            fz = utils.asarray(f, 'SurrogateFactory', name, dtype=float)
        
            if fz.ndim == 0:
                fz = np.full(self.nrand, fz)
            elif fz.shape != (self.nrand,):
                raise RuntimeError(f"SurrogateFactory: expected '{name}' constructor arg to be either"
                                   + f" shape ({self.nrand},) or a scalar (got shape {fz.shape})")

        assert fz.shape == (self.nrand,)
        assert fz.dtype == float

        if non_negative and np.min(fz) < 0:
            raise RuntimeError(f"SurrogateFactory: expected {name} to be >= 0")

        return fz
        

    def _eval_zarr(self, arr, name):
        if arr is None:
            return None

        arr = utils.asarray(arr, 'SurrogateFactory', name)
        
        if arr.shape == (self.nrand,):
            return arr
        
        raise RuntimeError(f"SurrogateFactory: expected '{name}' constructor arg to have shape ({self.nrand},), got shape {arr.shape}")


    def _check_ksz_args(self):
        if not self.ksz:
            for k in [ 'bvr', 'vr_noise_realization', 'vr_rms' ]:
                if getattr(self,k) is not None:
                    print(f"SurrogateFactory: warning: '{k}' constructor arg was specified, without also specifying ksz=True")
            return

        if self.bvr is None:
            raise RuntimeError("SurrogateFactory: if ksz=True is specified, then 'bvr' constructor arg must also be specified")
        if (self.vr_noise_realization is None) and (self.vr_rms is None):
            raise RuntimeError("SurrogateFactory: if ksz=True is specified, then 'vr_noise_realization' or 'vr_rms' constructor arg must also be specified")
        if (self.vr_noise_realization is not None) and (self.vr_rms is not None):
            raise RuntimeError("SurrogateFactory: specifying both 'vr_normalization' and 'vr_rms' is not allowed")
        
        
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
