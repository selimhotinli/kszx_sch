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
    def __init__(self, box, cosmo, ngal, bg, randcat, rweights=1.0, photometric=None, fnl=0, fsurr=1, ksz=False, bv=None, kernel='cubic'):
        """
        rweights: either None, or a 1-d array of length randcat.size (e.g. FKP weights).

        randcat: expect one of two cases
           1. spectroscopic: randcat contains 'z' column, but not 'ztrue' or 'zobs'.
           2. photometric: randcat contains 'ztrue' and 'zobs' columns, but not 'z'.
    
        ngal: only used to assign normalizations.

        bg: galaxy bias, can be either a scalar (constant galaxy bias), or a function z -> bg(z).
        """

        assert isinstance(box, Box)
        assert isinstance(cosmo, Cosmology)
        assert isinstance(randcat, Catalog)
        assert 0 < ngal < randcat.size
        assert 0 < fsurr <= 1
        assert fnl == 0    # placeholder
        assert not ksz     # placeholder
        assert bv is None  # placeholder

        self.box = box
        self.cosmo = cosmo
        self.ngal = ngal
        self.kernel = kernel
        self.nrand = randcat.size
        self.nsurr = int(fsurr * self.nrand + 0.5)
        self.fnl = fnl
        self.ksz = ksz
        # Note: don't need to save reference to 'randcat'.

        # Initialize ztrue_col, zobs_col
        # Determine whether survey is spectroscopic or photometric (see docstring).
        zcols = tuple([ (x in randcat.col_names) for x in ['z','ztrue','zobs'] ])

        if zcols == (True,False,False):
            if photometric is None:
                print("SurrogateFactory: setting photometric=False (catalog contains 'z' but not ztrue/zobs)")
                photometric = False
            elif photometric:
                raise RuntimeError("photometric=True was specified, but randcat only contains 'z' column")
            ztrue_col = 'z'
            zobs_col = 'z'
        elif zcols == (False,True,True):
            if photometric is None:
                print("SurrogateFactory: setting photometric=True (catalog contains ztrue/zobs but not 'z')")
                photometric = True
            elif not photometric:
                print("SurrogateFactory: photometric=False was specified, setting zobs=ztrue")
            ztrue_col = 'ztrue'
            zobs_col = 'zobs' if photometric else 'ztrue'
        else:
            raise RuntimeError("Expected one of two cases: (1) randcat contains 'z' but not ('ztrue','zobs'),"
                                + " or (2) randcat contains ('ztrue','zobs') but not 'z'. Specified randcat"
                                + f" contains the following columns: {sorted(randcat.col_names)}")

        rcat_ztrue = getattr(randcat, ztrue_col)
        self.rcat_xyz_true = randcat.get_xyz(cosmo, zcol_name = ztrue_col)
        self.rcat_xyz_obs = randcat.get_xyz(cosmo, zcol_name = zobs_col)
        
        # Initialize rcat_bg (two cases, since 'bg' can be either a function or a scalar).
        if callable(bg):
            rcat_bg = bg(rcat_ztrue)    # Note ztrue here
            rcat_bg = np.asarray(rcat_bg, dtype=float)
            if rcat_bg.shape != (self.nrand,):
                raise RuntimeError(f"Return value from bg() has shape {rcat_bg.shape}, expected shape {(self.nrand,)}")
        else:
            rcat_bg = np.asarray(bg, dtype=float)
            if rcat_bg.ndim != 0:
                raise RuntimeError(f"Expected 'bg' arg to be either a function or a scalar (got shape {rcat_bg.shape})")

        self.photometric = photometric
        self.sigma2 = self._integrate_kgrid(self.box, cosmo.Plin_z0(self.box.get_k()))
        self.rcat_bD = rcat_bg * cosmo.D(z=rcat_ztrue, z0norm=True)   # Note ztrue here
        self.rcat_wt = utils.asarray(rweights, 'SurrogateFactory', 'rweights', dtype=float)
        assert (self.rcat_wt.ndim == 0) or (self.rcat_wt.shape == (self.nrand,))

        v = self.sigma2 * self.rcat_bD**2
        self.fsurr_min = (ngal/self.nrand * np.max(v)) + 1.0e-10

        if self.fsurr_min > 1:
            nrand_min = int(self.fsurr_min * self.rand) + 1
            raise RuntimeError(f'SurrogateFactory: not enough randoms! (nrand={self.nrand}, {nrand_min=}')

        if fsurr < self.fsurr_min:
            raise RuntimeError(f'SurrogateFactor: fsurr is too small ({fsurr=}, fsurr_min={self.fsurr_min})')

        # If fsurr >= self.fsurr_min, then argument to sqrt(...) is always >= 0.
        self.rcat_eta_rms = np.sqrt(self.nsurr/ngal - v)

    
    @staticmethod
    def _integrate_kgrid(box, kgrid):
        """Helper for constructor. Could be moved somewhere more general (e.g. kszx.core)."""

        assert kgrid.shape == box.fourier_space_shape
    
        w = np.full(box.fourier_space_shape[-1], 2.0)
        w[0] = 1.0
        w[-1] = 2.0 if (box.real_space_shape[-1] % 2) else 1.0
        w = np.reshape(w, (1,)*(box.ndim-1) + (-1,))

        # FIXME could be optimized
        ret = np.sum(w * kgrid) / box.box_volume
        return np.real(ret)
    

    def simulate(self):

        # Step 1. Set up surrogate catalog:
        #   surr_xyz_true, surr_xyz_obs, surr_bg, surr_D.

        if self.nsurr == self.nrand:
            surr_xyz_true = self.rcat_xyz_true
            surr_xyz_obs = self.rcat_xyz_obs
            surr_bD = self.rcat_bD
            surr_wt = self.rcat_wt
            surr_eta_rms = self.rcat_eta_rms
        else:
            isrc = np.random.permutation(self.nrand)[:self.nsurr]
            surr_xyz_true = self.rcat_xyz_true[isrc]
            surr_xyz_obs = self.rcat_xyz_obs[isrc]
            surr_bD = self.rcat_bD[isrc]
            surr_wt = self.rcat_wt[isrc] if (self.rcat_wt.ndim > 0) else self.rcat_wt
            surr_eta_rms = self.rcat_eta_rms[isrc]
        
        # Step 2. Evaluate fields:
        #   surr_delta0, surr_vr0, surr_phi0

        delta0 = simulate_gaussian_field(self.box, self.cosmo.Plin_z0)
        delta0 /= np.sqrt(compensation_kernel(self.box, self.kernel))

        if self.ksz:
            faH0 = self.cosmo.frsd(z=0) * self.cosmo.H(z=0)
            vr0 = multiply_kfunc(self.box, delta0, lambda k: faH0/k, dc=0)
            surr_vr0 = interpolate_points(self.box, vr0, surr_xyz_true, self.kernel, fft=True, spin=1)
            del vr0

        if self.fnl != 0:
            phi0 = multiply_kfunc(self.box, delta0, lambda k: 1.0/self.cosmo.alpha(k=k,z=0), dc=0)
            surr_phi0 = interpolate_points(self.box, phi0, surr_xyz_true, self.kernel, fft=True)
            del phi0

        surr_wg = surr_bD * interpolate_points(self.box, delta0, surr_xyz_true, self.kernel, fft=True)
        surr_wg += np.random.normal(scale = surr_eta_rms)
        surr_wg *= (self.ngal / self.nsurr) * surr_wt
        self.Sg = grid_points(self.box, surr_xyz_obs, surr_wg, kernel=self.kernel, fft=True)
        
