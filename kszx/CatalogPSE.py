# To do:
#  - make sure that DC mode is being included
#  - explore choice of K empirically in a few examples
#  - don't forget to think about compensation!!
#  - when the dust settles, create an interface for compensation_kernel which uses less memory.
#  - I guess I'm renaming kbin_delim -> kbin_edges
#  - Add helper functions to make (galaxies - randoms) plots?

from . import core
from . import utils

from collections.abc import Iterable


class CatalogPSE:
    def __init__(self, box, kbin_edges, rpoints, *, rweights=None, spin=0, nfields=1, kernel='cubic', use_dc=False, return_1d=True, heavyweight=None):
        assert isinstance(box, Box)
        assert int(nfields) == nfields
        assert nfields >= 1
        assert nfields <= 50
        
        # If use_dc=False, then core._check_kbin_delim() will change kmin.
        self.kbin_edges = core._check_kbin_delim(box, kbin_delim, use_dc)
        self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.
        self.return_1d = return_1d
        self.nfields = int(nfields)
        self.nkbins = len(self.kbin_edges) - 1
        self.box = box
        self.spin = self._parse_spin(spin)
        self.kernel = kernel
        self.heavyweight = heavyweight

        rpoints, rweights, _ = self._parse_pwv(rpoints, rweights, None, 'CatalogPSE.__init__()', prefix='r')

        # FIXME when the dust settles, create an interface for compensation_kernel which uses less memory.
        ck = 1.0 / np.sqrt(core.compensation_kernel(box, kernel))
        
        self.rmaps = [ None for _ in range(self.nfields) ]  # real space
        self.fmaps = [ None for _ in range(self.nfields) ]  # Fourier space
        
        for i in range(self.nfields):
            for j in range(i):
                if (rpoints[i] is rpoints[j]) and (rweights[i] is rweights[j]):
                    self.rmaps[i] = rmaps[j]
                    self.fmaps[i] = fmaps[j]
                    break
            else:
                self.rmaps[i] = core.grid_points(box, rpoints[i], rweights[i], kernel=kernel)
                self.fmaps[i] = core.fft_r2c(box, rmaps[i])
                self.fmaps[i] *= ck

        # Save the real-space maps, for later use in apply().
        # Note: saving Fourier-space maps doesn't work, since apply() can use nonzero spins.
        self.rmaps = rmaps

        if heavyweight:
            self.fmaps = fmaps

        print('FIXME placeholder value of K!!')
        self.normalization = self.compute_normalization(K=0.1)


    def apply(self, points, weights=None, values=None):
        """
        Case 1 (values is None): Density field of tracers. Randoms will be subtracted.

        Case 2 (values is not None): General sum of delta functions. No random subtraction.
        Examples: kSZ velocity reconstruction, surrogate field.
        
        Note: in the future I might add an 'rvalues' optional argument, to enable random subtraction
        in Case 2.
        """
        
        points, weights, values = self._parse_pwv(points, weights, values, 'CatalogPSE.apply()')

        # Fourier-space maps
        fmaps = [ None for _ in range(self.nfields) ]

        for i in range(self.nfields):
            p,w,v = points[i], weights[i], values[i]
            
            if v is None:
                # Case 1: density field of tracers, with randoms subtracted.
                m = core.grid_points(self.box, p, w, kernel=self.kernel)
                m -= (xx) * self.rmaps[i]   # subtract randoms
            else:
                # Case 2: general sum of delta functions, with no random subtraction.
                m = core.grid_points(self.box, p, w*v, kernel=self.kernel)

            fmaps[i] = core.fft_r2c(self.box, m, spin = self.spin[i])
            del m

        pk = core.estimate_power_spectrum(box, fmaps, self.kbin_edges, use_dc = self.use_dc)
        assert pk.shape == (self.nfields, self.nfields, self.nkbins)
        
        return pk[0,0,:] if (self.return_1d and (self.nfields == 1)) else pk


    def compute_normalization(self, K):
        """Returns a shape (self.nfields, self.nfields) array. Denoted A_{WW'} in the notes."""

        if not hasattr(self, 'fmaps'):
            raise RuntimeError(f'CatalogPSE: to call compute_normalization()')
            
        edges = np.array([ 0, K, 2**(1./self.box.ndim) * K ])

        # use_dc=True is important here!!
        pk, counts = core.estimate_power_spectrum(self.box, self.fmaps, edges, use_dc=True, return_counts=True)
        assert pk.shape == (self.nfields, self.nfields, 2)
        assert counts.shape == (2,)

        return (pk[:,:,0]*counts[0] - pk[:,:,1]*counts[1])
        

    ################################################################################################


    def _check_length(self, l, caller, argname):
        assert isinstance(l, list)
        
        if len(l) == 1:
            return [ l[0] for _ in range(self.nfields) ]
        if len(l) == self.nfields:
            return l

        s = f'1 or {self.nfields} [=nfields]' if (self.nfields > 1) else '1'
        
        raise RuntimeError(f"{caller}: expected '{argname}' to have length {s}, got length {len(l)}. This"
                           + f" error can occur if you forgot to pass 'nfields' to the CatalogPSE constructor.")
        
    
    def _parse_points(self, points, caller, argname):
        """Return value is a length-nfields list of float arrays, but arrays are not shape-checked."""
        
        if not isinstance(points, Iterable):
            raise RuntimeError(f"{caller}: expected '{argname}' to be a list or array, got type {type(points)}")
        
        if not isinstance(points, np.ndarray):
            points = [ utils.asarray(a,caller,argname,float) for a in points ]
            return self._check_length(points)

        points = utils.asarray(points, caller, argname, float)
        
        if (points.ndim == 2) and (points.shape[1] == self.box.ndim):
            return [ points for _ in range(self.nfields) ]
        if (points.ndim == 3) and (points.shape[0] == self.nfields) and (points.shape[2] == self.box.ndim):
            return self._check_length(list(points))
        
        raise RuntimeError(f"{caller}: expected {argname}.shape = ({self.nfields},*,{self.box.ndim}) or (*,{self.box.ndim}),"
                           + f" got shape {points.shape}. (Note that nfields={self.nfields} and box.ndim={self.box.ndim}."
                           + f" This error can occur if you forgot to pass 'nfields' to the CatalogPSE constructor.)")

    
    def _parse_vals(self, vals, caller, argname):
        """Return value is a length-nfields list of (float array or None), but arrays are not shape-checked."""

        if vals is None:
            return [ None for _ in range(self.nfields) ]
        
        if not isinstance(vals, Iterable):
            raise RuntimeError(f"{caller}: expected '{argname}' to be list/array/None, got type {type(vals)}")
        
        if not isinstance(vals, np.ndarray):
            vals = [ utils.asarray(a, caller, argname, float, allow_none=True) for a in vals ]
            return self._check_length(vals)

        vals = utils.asarray(vals, caller, argname, float)
            
        if vals.ndim == 1:
            return [ vals  for _ in range(self.nfields) ]
        if (vals.ndim == 2) and (vals.shape[0] == self.nfields):
            return list(vals)
        
        raise RuntimeError(f"{caller}: expected '{argname}' to be a 1-d array or have shape ({self.nfields},*),"
                           + f" got shape {vals.shape}. (Note that nfields={self.nfields}. This error can occur"
                           + " if you forgot to pass 'nfields' to the CatalogPSE constructor.")

    
    def _parse_pwv(self, points, weights, values, caller, prefix=''):
        points = self._parse_points(points, caller, f'{prefix}points')
        weights = self._parse_vals(weights, caller, f'{prefix}weights')
        values = self._parse_vals(values, caller, f'{prefix}values')

        assert len(points) == len(weights) == len(values) == self.nfields

        # Shape checks.
        for p,w,v in zip(points, weights, values):
            if (p.ndim != 2) or (p.shape[1] != self.box.ndim):
                raise RuntimeError(f"{caller}: expected {prefix}points to have shape (*,{self.box.ndim}), got shape {p.shape}")
            
            if w is not None:
                if w.ndim != 1:
                    raise RuntimeError(f"{caller}: expected {prefix}weights to be a 1-d array, got shape {w.shape}")
                if p.shape[0] != w.shape[0]:
                    raise RuntimeError(f"{caller}: {prefix}points/{prefix}weights arrays have unequal lengths ({p.shape[0]}, {w.shape[0]})")
                if np.min(w) < 0:
                    raise RuntimeError(f"{caller}: {prefix}weights arrays have unequal lengths ({p.shape[0]}, {w.shape[0]})")                

            if v is not None:
                if v.ndim != 1:
                    raise RuntimeError(f"{caller}: expected {prefix}values to be a 1-d array, got shape {v.shape}")
                if p.shape[0] != v.shape[0]:
                    raise RuntimeError(f"{caller}: {prefix}points/{prefix}values arrays have unequal lengths ({p.shape[0]}, {v.shape[0]})")
            
        return points, weights, values
        

    def _parse_spin(self, spin):
        spin = utils.asarray(spin, 'CatalogPSE.__init__()', 'spin', int)

        if spin.ndim == 0:
            spin = np.full(self.nfields, spin, dtype=int)
        elif spin.shape == (1,):
            spin = np.full(self.nfields, spin[0], dtype=int)
        elif spin.shape != (self.nfields,):
            raise RuntimeError("CatalogPSE.__init__(): expected 'spin' argument to be a scalar or length-{self.nfields}, got shape {spin.shape}")

        if np.min(spin) < 0:
            raise RuntimeError("CatalogPSE.__init__(): spins must be non-negative")
        if np.max(spin) > 1:
            raise RuntimeError("CatalogPSE.__init__(): currently, only spin=0 and spin=1 are supported")

        return spin
