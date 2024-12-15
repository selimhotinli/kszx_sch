import numpy as np

from .Box import Box
from .Catalog import Catalog
from .Cosmology import Cosmology

from . import utils

from .core import \
    interpolate_points, \
    grid_points, \
    compensation_kernel, \
    multiply_kfunc, \
    simulate_gaussian_field


class SurrogateFactory:
    def __init__(self, box, cosmo, randcat, ngal, bg, fnl=0, ksz=False, bv=None, sigmav=None, photometric=None, deltac=1.68, kernel='cubic'):
        """
        randcat: expect one of two cases
           1. spectroscopic: randcat contains 'z' column, but not 'ztrue' or 'zobs'.
           2. photometric: randcat contains 'ztrue' and 'zobs' columns, but not 'z'.
    
        ngal: only used to assign normalizations.

        The bg, bv, and sigmav arguments represent galaxy bias, kSZ velocity reconstruction bias, and
        kSZ velocity reconstruction RMS noise. They can be any of the following:

          1. a function z -> b(z)
          2. a 1-d array of length randcat.size
          3. a scalar

        photometric: if None, will autodetect from randcat.
        kernel: currently, either 'cic' or 'cubic' are supported.
        """

        assert isinstance(box, Box)
        assert isinstance(cosmo, Cosmology)
        assert isinstance(randcat, Catalog)
        assert 0 < ngal < randcat.size
            
        self.box = box
        self.cosmo = cosmo
        self.ngal = ngal
        self.kernel = kernel
        self.nrand = randcat.size
        self.fnl = fnl
        self.ksz = ksz
        self.deltac = deltac
        # Note: don't need to save reference to 'randcat'.

        rcat_ztrue, rcat_zobs = self._init_redshifts(randcat, photometric)
        bg = self._eval_zfunc(b, rcat_ztrue, 'bg')
        D = cosmo.D(z=rcat_ztrue, z0norm=True)
        
        self.rcat_xyz_true = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=rcat_ztrue))
        self.rcat_xyz_obs = utils.ra_dec_to_xyz(randcat.ra_deg, randcat.dec_deg, r=cosmo.chi(z=rcat_zobs))
        self.sigma2 = self._integrate_kgrid(self.box, cosmo.Plin_z0(self.box.get_k()))
        self.rcat_bD = bg * D
        self.nrand_min = int(ngal * self.sigma2 * np.max(self.rcat_bD)**2 + 1.01)

        if self.nrand < self.nrand_min:
            raise RuntimeError(f'SurrogateFactory: not enough randoms! (nrand={self.nrand}, nrand_min={self.nrand_min})')

        # If nrand >= self.nrand_min, then argument to sqrt(...) is always >= 0.
        self.rcat_sigmag = np.sqrt(self.nrand/ngal - (self.sigma2 * self.rcat_bD**2))

        if ksz:
            self.rcat_sigmav = self._eval_zfunc(sigmav, rcat_ztrue, 'sigmav')
            self.rcat_bv_faHD = self._eval_zfunc(bv, rcat_ztrue, 'bv')
            self.rcat_bv_faHD *= cosmo.frsd(z=rcat_ztrue) * cosmo.H(z=rcat_ztrue) * D / (1 + rcat_ztrue)

        if fnl != 0:
            self.rcat_bng = deltac * (bg-1)


    def _init_redshifts(self, randcat, photometric):
        """Helper method called by constructor. Returns (rcat_ztrue, rcat_zobs). 

        The 'photometric' arg can be None (for "autodetect from randcat")."""

        zcols = set(['z','ztrue','zobs'])
        zcols = zcols.intersection(randcat.col_names)

        # Case 1: spectroscopic catalog.
        if zcols == set(['z']):
            if photometric is None:
                print("SurrogateFactory: setting photometric=False (randcat contains 'z' but not ztrue/zobs)")
            elif photometric:
                raise RuntimeError("photometric=True was specified, but randcat only contains 'z' column")
            return randcat.z, randcat.z

        # Case 2: photometric catalog.
        if zcols == set(['ztrue','zobs']):
            if photometric is None:
                print("SurrogateFactory: setting photometric=True (catalog contains ztrue/zobs but not 'z')")
            elif not photometric:
                print("SurrogateFactory: randcat is photometric, but photometric=False was specified,"
                      " setting zobs=ztrue to mimic spectroscopic catalog")
                return randcat.ztrue, randcat.ztrue
            else:
                return randcat.ztrue, randcat.zobs

        raise RuntimeError("Expected one of two cases: (1) spectroscopic randcat containing 'z' but not ('ztrue','zobs'),"
                           + " or (2) photometric randcat containing ('ztrue','zobs') but not 'z'. Actual randcat contains"
                           + f" the following columns: {sorted(randcat.col_names)}")

        
    def _eval_zfunc(self, b, z, name):
        """Helper method for contructor. Returns an array of length self.nrand.
        The 'b' argument is either a callable function z -> b(z), an array of length self.nrand, or a scalar.
        """

        if b is None:
            raise RuntimeError(f"SurrogateFactory: '{b}' constructor argument must be specified")
        
        if callable(b):
            bz = b(z)
            bz = np.asarray(bz, dtype=float)
            if bz.shape != (self.nrand,):
                raise RuntimeError(f"Return value from {b}() has shape {bz.shape}, expected shape {(self.nrand,)}")
            return bz

        bz = np.asarray(bz, dtype=float)
        
        if bz.ndim == 0:
            return np.full(self.nrand, bz)
        if bz.shape == (self.nrand,):
            return bz

        raise RuntimeError(f"SurrogateFactory: expected '{b}' constructor arg to be either shape ({self.nrand},)"
                           + f" or a scalar (got shape {bz.shape})")

        
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


    def simulate(self):
        """Initializes self.surr_deltag, self.surr_vr0 (1-d arrays of length self.nrand)."""

        delta0 = simulate_gaussian_field(self.box, self.cosmo.Plin_z0)
        delta0 /= np.sqrt(compensation_kernel(self.box, self.kernel))
        
        self.surr_deltag = self.rcat_bD * interpolate_points(self.box, delta0, self.rcat_xyz_true, self.kernel, fft=True)
        self.surr_deltag += np.random.normal(scale = self.rcat_sigmag)
        self.surr_deltag *= (self.ngal / self.nsurr)

        if self.ksz:
            vr0 = multiply_kfunc(self.box, delta0, lambda k: self.rcat_faH0 / k, dc=0)
            surr_vr = self.rcat_bv_faHD * interpolate_points(self.box, vr0, self.rcat_xyz_true, self.kernel, fft=True, spin=1)
            del vr0

        if self.fnl != 0:
            phi0 = multiply_kfunc(self.box, delta0, lambda k: 1.0/self.cosmo.alpha(k=k,z=0), dc=0)
            surr_phi0 = interpolate_points(self.box, phi0, self.rcat_xyz_true, self.kernel, fft=True)
            del phi0


        
