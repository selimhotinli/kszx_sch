r"""CatalogPSE: a power spectrum estimator for weighted catalogs (including cross-spectra).

The main functions are __init__() and apply(). You might start by reading the cookbook
below, and then the docstrings for __init__() and apply().

Should be general enough to apply to any field which can be represented as a weighted
sum of delta functions, including galaxy overdensity fields, kSZ velocity reconstruction,
and surrogate fields. Designed to work with class :class:`~kszx.SurrogateFactory`.

Uses an approximate prescription for the power spectrum normalization, such that power
spectra like $P_{gg}$ and $P_{gv}$ should agree with CAMB to say 10%, except perhaps at
low k, where there may be some artifacts. A more accurate normalization is coming soon!
In the meantime, if you need accurate normalization, you should compare "data" power
spectra to surrogate fields (see :class:`~kszx.SurrogateFactory`).

CatalogPSE cookbook
-------------------

We'll explain the CatalogPSE using the following two examples (throughout, we use
notation from the overleaf):

 - Case 1 (data). KSZ velocity reconstruction analysis, with two fields $\delta_g, \hat v_r$
   of the form
  
   $$\begin{align}
   \delta_g(x) &\propto \sum_{i \in gal} W^L_i \delta^3(x-x_i) - \frac{N_{gal}}{N_{rand}} \sum_{j \in rand} W^L_j \delta^3(x-x_j) \\
   \hat v_r(x) &= \sum_{i \in gal} W^S_i T_{CMB}(\theta_i) \delta^3(x-x_i)
   \end{align}$$

   Following the overleaf, we have included per-object weightings (denoted $W^L_i$, $W^S_i$ 
   for $\delta_g$ and $\hat v_r$ respectively). These could be FKP weightings or similar.

 - Case 2. Surrogate fields (probably created with :class:`~kszx.SurrogateFactory`) $S_g, S_v$,
   used to assign error bars to the KSZ analysis from case 1.

   $$\begin{align}
   S_g(x) &\propto \frac{N_{gal}}{N_{rand}} \sum_{j \in rand} W_j^L (\delta_G(x_j) + \mbox{noise}) \delta^3(x-x_j) \\
   S_v(x) &= \frac{N_{gal}}{N_{rand}} \sum_{j \in rand} W_j^S (b_j^v v_r(x_j) + \mbox{noise}) \delta^3(x-x_j)
   \end{align}$$

   See overleaf for definitions of $\delta_G, b_j^v$, etc.

We'll construct a ``CatalogPSE`` object once, and then call ``apply()``  once per evaluation 
of the power spectrum estimator. The details are mostly self-evident, except for three function
arguments: the ``rweights`` argument to the constructor, and the ``weights``, ``values`` 
arguments to ``apply()``.

Here's the "cookbook" for cases 1+2, which also demonstrates how to use :class:`~kszx.SurrogateFactory`::

 # Setup: galaxy catalog (used in case 1).
 gal_xyz = ...   # shape-(ngal,3) array with 3-d galaxy locations
 gal_wl = ...    # shape-(ngal,) array with weights W^L_i (see above)
 gal_ws = ...    # shape-(ngal,) array with weights W^S_i (see above)
 gal_bv = ...    # shape-(ngal,) array with vrec bias b_i^v (see above)
 gal_tcmb = ...  # shape-(ngal,) array with filtered CMB temperatures

 # Setup: random catalog (used in cases 1+2).
 rand_xyz = ...  # shape-(nrand,3) array with 3-d locations of randoms
 rand_wl = ...   # shape-(nrand,) array with weights W^L_i (see above)
 rand_ws = ...   # shape-(nrand,) array with weights W^S_i (see above)
 rand_bv = ...   # shape-(nrand,) array with vrec bias b_i^v
 rand_tcmb = ... # shape-(nrand,) array with filtered CMB temperatures

 # Setup: power spectrum estimator (used in cases 1+2).
 pse = kszx.CatalogPSE(
   box, kbin_edges,      # see "constructor arguments" below
   rpoints = rand_xyz,   # defines survey footprint
   rweights = [rand_wl, rand_ws * rand_bv],  # defines response of delta_g, v_r fields
   spin = [0,1], # super important: vr has spin 1!!
   nfields = 2)

 # Case 1: KSZ velocity reconstruction analysis with delta_g, v_r.
 #
 # CatalogPSE.apply() returns a shape-(2,2,nkbins) array.
 # The first two indices are [[Pgg,Pgv],[Pgv,Pvv]] in an approximate normalization.
 # Note that values=None for the first field (delta_g). This means that delta_g is
 # treated as an overdensity field, and random subtraction is performed.

 pk = pse.apply(
   points = gal_xyz,
   weights = [gal_wl, gal_ws * gal_bv],
   values = [None, gal_ws * gal_tcmb]
 )

 # Case 2: Surrogate fields S_g, S_v (in order to assign error bars).

 # We use the SurrogateFactory helper class to simulate S_g, S_v.
 #
 # Simulating the velocity reconstruction noise is nontrivial. One approach
 # is to use the same noise realization for all surrogate fields, obtained
 # directly from the data as (rand_ws * rand_tcmb), or (W_S^j * T_CMB^j) in
 # notation from the overleaf. For more info on 'class SurrogateFactory',
 # see its docstring.

 sf = kszx.SurrogateFactory(
    ...,   # not all constructor arguments are shown
    gweights = rand_wl,       # used for S_g
    ksz_gweights = rand_ws,   # used for S_v
    ksz_tcmb_realization = rand_tcmb,  # used to "bootstrap" reconstruction noise
    ksz_bv = rand_bv,
    ksz = True
 )

 # To simulate S_g, S_v, we call SurrogateFactory.simulate(), which initializes
 # (see above for notation):
 # 
 #   sf.surr_gal_weights    [ = (Ngal/Nrand) W_j^L ] 
 #   sf.surr_gal_values     [ = (Ngal/Nrand) W_j^L (\delta_G(x_j) + \eta_j) ]
 #   sf.surr_vr_weights     [ = (Ngal/Nrand) W_j^S b_j^v ]
 #   sf.surr_vr_values      [ = (Ngal/Nrand) W_j^S (b_j^v v_r(x_j) + noise) ]
 #
 # These quantities will be passed to CatalogPSE.apply() in the next step.

 sf.simulate()

 # CatalogPSE.apply() returns a shape-(2,2,nkbins) array.
 # The first two indices are [[Pgg,Pgv],[Pgv,Pvv]] in an approximate normalization.

 pk = pse.apply(
   points = sf.xyz_obs,   # use observed (not true) redshfts, if catalog is photometric
   weights = [sf.surr_gal_weights, sf.surr_vr_weights],
   values = [sf.surr_gal_values, sf.surr_vr_values]
 )

 # (In a real pipeline, the calls to sf.simulate() and pse.apply() would be
 # repeated in a loop, in order to analyze many surrogate sims.)

Some notes on the cookbook:

  - In the KSZ case, set rweights=bv (note that bv includes W_S) in the constructor.
    When calling apply() in case 1, set weights=bv and values=ws*tcmb.
    (In case 2, the SurrogateFactory automatically sets up the weights/values.)

  - Don't forget ``spin=[0,1]`` in the CatalogPSE constructor!
    If this argument is omitted, then vr will be treated as a scalar field, which
    will (silently) produce incorrect results.

  - In this example, we used the same random catalog for delta_g and v_r,
    but it's possible to use different random catalog (e.g. one could use
    SDSS for delta_g and DESILS-LRG for v_r).

  - Consistency of randcats is important! (Elaborate.)

  - SurrogateFactory and CatalogPSE are designed to work together.

What CatalogPSE really computes
-------------------------------

You probably don't need to read this -- I just found it helpful to leave notes
to myself! Here's a precise specification of what CatalogPSE.appy() computes.
We'll give a formal definition first, and then motivate the definition with 
some examples:

  - When CatalogPSE is constructed, the per-field randoms are specified by 
    ``rpoints`` $x_j^{rand}$ and ``rweights`` $W_j^{rand}$. Let $R(x)$ be the
    random field:

    $$R(x) = \sum_{j\in rand} W_j^{rand} \delta^3(x-x_j)$$
     
  - When CatalogPSE.simulate() is called, each field is represented by ``points``
    $x_i$, ``weights`` $W_i$, and ``values`` $V_i$. (Note that the values array can
    be None.) Define $W_{tot}$ and $W_{rtot}$ by:

    $$W_{tot} = \sum_i W_i \hspace{1cm} W_{rtot} = \sum_{j\in rand} W_j^{rand}$$

    Then, we define a real-space field $f(x)$, with two cases as follows.

  - If ``values`` is None, then the ``points`` $x_i$ and ``weights`` $W_i$ are 
    intepreted as a tracer density field. We define:

    $$\begin{align}
    f(x) &= \left( \sum_i W_i \delta^3(x-x_i) \right) - \frac{W_{tot}}{W_{rtot}} \left( \sum_{j\in rand} W_j \delta^3(x-x_j) \right) \\
    {\mathcal N} &= \frac{W_{tot}}{W_{rtot}}
    \end{align}$$

  - If ``values`` is not None, then the ``points`` $x_i$, ``weights`` $W_i$, and 
    ``values`` $V_i$ are interpreted as a catalog onto which values have been 
    "painted". We define:

    $$\begin{align}
    f(x) &= \sum_i V_i \delta^3(x-x_i) \\
    {\mathcal N} &= \frac{W_{tot}}{W_{rtot}}
    \end{align}$$

  - Consider two real-space fields $f(x)$, $f'(x)$ with normalizations ${\mathcal N}$,
    ${\mathcal N}'$, defined by the above bullet points. The "normalized" power 
    spectrum returned by CatalogPSE.apply() is:

    $$P_{ff'}^{norm}(k) = \frac{1}{{\mathcal N} {\mathcal N}'} P_{ff'}^{nrand}(k)$$

    where $P_{ff'}^{nrand}(k)$ is a power spectrum estimator, to be constructed in
    detail below, which is "normalized to randoms" in the following sense.
    Suppose that the fields $f,f'$ were obtained by multiplying "unwindowed" 
    fields $F,F'$ by the random catalogs $R,R'$:

    $$\begin{align}
    f(x) &= R(x) F(x) = \sum_{j\in rand} W_j F(x_j) \delta^3(x-x_j)  \\
    f'(x) &= R'(x) F'(x) = \sum_{j\in rand'} W'_j F(x'_j) \delta^3(x-x'_j)
    \end{align}$$

    Then, the estimator $P^{nrand}_{ff'}(k)$ has (approximately) 
    the same normalization as the unwindowed power spectrum $P_{FF'}(k)$.

We'll define $P^{nrand}_{ff'}(k)$ shortly, but first let's consider some examples.
In the cookbook above (cases 1+2), our calls to CatalogPSE.apply() implicitly
define fields $\delta_g$, $\hat v_r$, $S_g$, $S_v$, and our call to 
CatalogPSE.__init__() defines fields $R_g$, $R_v$. Using the specification
in the previous bullet points, we can write down all these fields precisely:

    $$\begin{align}
    \delta_g(x)
    &    = \left( \sum_i W_i^L \delta^3(x-x_i) \right) - \left( \frac{W_{tot}^L}{W_{rtot}^L} \sum_{j\in rand} W_j^L \delta^3(x-x_j) \right)  \\
    \hat v_r(x) 
    &    = \sum_i W_i^S T_{cmb}(\theta_i) \delta^3(x-x_i) \\
    S_g(x)
    &    = \frac{N_g}{N_r} \sum_{j\in rand} W_j^L (b_j^g \delta_m(x_j) + \mbox{noise}) \delta^3(x-x_j) \\
    S_v(x)
    &    = \frac{N_g}{N_r} \sum_{j\in rand} W_j^S (b_j^v v_r(x_j) + \mbox{noise}) \delta^3(x-x_j) \\
    R_g(x) 
    &    = \sum_{j\in rand} W_j^L \delta^3(x-x_j) \hspace{1.6cm} (\mbox{randcat for $\delta_g$, $S_g$}) \\
    R_v(x) 
    &    = \sum_{j\in rand} W_j^S b_j^v \delta^3(x-x_j) \hspace{1.3cm} (\mbox{randcat for $\hat v_r$, $S_v$}) \\
    {\mathcal N}
    &    \approx \frac{N_g}{N_r} \hspace{3cm} (\mbox{for all fields: } \delta_g, \hat v_r, S_g, S_v)
    \end{align}$$

To conclude this section, it remains to construct the power spectrum estimator
$P^{nrand}_{ff'}(k)$. Let $f,f'$ be a pair of fields, and let $R,R'$ be the
associated random fields (see above).
Let $P_{RR'}(k)$ be the (unnormalized) power spectrum in volume $V_{box}$.
Define:

$$A_{RR'} \equiv \left(\int_{k < 2^{1/3}K} - \int_{2^{1/3}K < k < K} \right) 
\frac{d^3k}{(2\pi)^3} P_{RR'}(k)$$

The purpose of the subtraction is to cancel shot noise.
Then, we define $P^{nrand}_{ff'}(k)$ by:

$$P^{nrand}_{ff'}(k) = \frac{1}{A_{RR'}} P^{raw}_{ff'}(k)$$

To get some intuition for what $A_{RR'}$ represents, suppose all weights have
constant values $W,W'$, and the randcats have number densities $n,n'$ in 
volumes $V,V'$. Then:

$$A_{RR'} \approx nn'WW' \frac{V \cap V'}{V_{\rm box}}$$

Class description
-----------------
"""

from . import core
from . import utils

from .Box import Box

import numpy as np
from collections.abc import Iterable


class CatalogPSE:
    def __init__(self, box, kbin_edges, rpoints, *, rweights=None, spin=0, nfields=1, kernel='cubic', compensate=True, use_dc=False, return_1d=True, heavyweight=None):
        """
        Constructor arguments:
        
           - ``rpoints``: probably obtained by calling Catalog.get_xyz().
           - ``rweights``: if None, default is all-ones.
           - ``nfields``: number of 
        """
        
        assert isinstance(box, Box)
        assert int(nfields) == nfields
        assert nfields >= 1
        assert nfields <= 50
        
        # If use_dc=False, then core._check_kbin_delim() will change kmin.
        self.box = box
        self.kbin_edges = core._check_kbin_delim(box, kbin_edges, use_dc)
        self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.
        self.nkbins = len(self.kbin_edges) - 1
        self.nfields = int(nfields)
        self.spin = self._parse_spin(spin)
        self.kernel = kernel
        self.compensate = compensate
        self.use_dc = use_dc
        self.return_1d = return_1d
        self.heavyweight = heavyweight

        # rpoints = length-nfields list of shape-(nrand,box.ndim) arrays
        # rweights = length-nfields list of (shape-(nrand,) array or None)
        # wrtot = numpy array of shape (nfields,)
        rpoints, rweights, _, wrtot = self._parse_pwv(rpoints, rweights, None, 'CatalogPSE.__init__()', prefix='r')

        # FIXME when the dust settles, create an interface for compensation_kernel which uses less memory.
        self.ck = 1.0 / np.sqrt(core.compensation_kernel(box, kernel))
        
        self.rmaps = [ None for _ in range(self.nfields) ]  # real space maps
        self.fmaps = [ None for _ in range(self.nfields) ]  # Fourier space maps
        self.wrtot = wrtot
        
        for i in range(self.nfields):
            # (rpoints, rweights) pairs can be repeated -- in this case we can save memory.
            for j in range(i):
                if (rpoints[i] is rpoints[j]) and (rweights[i] is rweights[j]):
                    self.rmaps[i] = self.rmaps[j]
                    self.fmaps[i] = self.fmaps[j]
                    break
            else:
                self.rmaps[i] = core.grid_points(box, rpoints[i], rweights[i], kernel=kernel)
                self.fmaps[i] = core.fft_r2c(box, self.rmaps[i])
                self.fmaps[i] *= ck
            
        # FIXME revisit the issue of choosing K!
        K = 0.6 * box.knyq
        self.A = self.compute_A(K)
        assert self.A.shape == (self.nfields, self.nfields)
        assert np.all(self.A.diagonal() > 0)

        self.Ainv = np.zeros((self.nfields, self.nfields))
        
        for i in range(self.nfields):
            for j in range(i+1):
                r = self.A[i,j] / np.sqrt(self.A[i,i] * self.A[j,j])
                if r < 0.03:
                    print(f'CatalogPSE: fields {(j,i)} have non-overlapping footprints, cross power will not be estiamted')
                else:
                    self.Ainv[i,j] = self.Ainv[j,i] = 1.0 / self.A[i,j]

        # We save the real-space maps, for later use in apply().
        # (Note: saving Fourier-space maps doesn't work, since apply() can use nonzero spins.)
        # If heavyweight=True, then we also save Fourier-space maps, for use in compute_A().

        if not heavyweight:
            del self.fmaps


    def apply(self, points, weights=None, values=None):
        """
        Case 1 (values is None): Density field of tracers. Randoms will be subtracted.

        Case 2 (values is not None): General sum of delta functions. No random subtraction.
        Examples: kSZ velocity reconstruction, surrogate field.
        
        Note: in the future I might add an 'rvalues' optional argument, to enable random subtraction
        in Case 2.
        """
        
        points, weights, values, wtot = self._parse_pwv(points, weights, values, 'CatalogPSE.apply()')
        wratios = wtot / self.wrtot

        # Fourier-space maps (initialized in loop)
        fmaps = [ None for _ in range(self.nfields) ]

        for i in range(self.nfields):
            p,w,v = points[i], weights[i], values[i]
            
            if v is None:
                # Case 1: density field of tracers, with randoms subtracted.
                m = core.grid_points(self.box, p, w, kernel=self.kernel)
                m -= wratios[i] * self.rmaps[i]   # subtract randoms
            else:
                # Case 2: general sum of delta functions, with no random subtraction.
                m = core.grid_points(self.box, p, v, kernel=self.kernel)

            fmaps[i] = core.fft_r2c(self.box, m, spin = self.spin[i])
            del m

            if self.compensate:
                fmaps[i] *= self.ck

        pk = core.estimate_power_spectrum(box, fmaps, self.kbin_edges, use_dc = self.use_dc)
        assert pk.shape == (self.nfields, self.nfields, self.nkbins)

        pk /= np.reshape(wratios, (self.nfields, 1, 1))
        pk /= np.reshape(wratios, (1, self.nfields, 1))
        pk *= np.reshape(self.Ainv, (self.nfields, self.nfields, 1))

        if self.return_1d and (self.nfields == 1):
            pk = pk[0,0,:]
            
        return pk


    def compute_A(self, K):
        """Returns a shape (self.nfields, self.nfields) array. Denoted A_{WW'} in the notes."""

        if not hasattr(self, 'fmaps'):
            raise RuntimeError('CatalogPSE: in order to call compute_A(), you'
                               + ' should call the constructor with heavyweight=True')
            
        edges = np.array([ 0, K / 2**(1./self.box.ndim), K ])

        # use_dc=True is important here!!
        pk, counts = core.estimate_power_spectrum(self.box, self.fmaps, edges, use_dc=True, return_counts=True)
        assert pk.shape == (self.nfields, self.nfields, 2)
        assert counts.shape == (2,)

        # Note factor (1/V_box), which normalizes sum_k -> int d^3k/(2pi)^3
        return (pk[:,:,0] - pk[:,:,1]) * counts[0] / self.box.box_volume
        

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
        wtot = np.zeros(self.nfields)

        # Shape checks.
        for i in range(self.nfields):
            p,w,v = points[i], weights[i], values[i]

            if (p.ndim != 2) or (p.shape[1] != self.box.ndim):
                raise RuntimeError(f"{caller}: expected {prefix}points to have shape (*,{self.box.ndim}), got shape {p.shape}")
            if p.shape[0] == 0:
                raise RuntimeError(f"{caller}: {prefix}points array has zero length")

            if w is not None:
                if w.ndim != 1:
                    raise RuntimeError(f"{caller}: expected {prefix}weights to be a 1-d array, got shape {w.shape}")
                if p.shape[0] != w.shape[0]:
                    raise RuntimeError(f"{caller}: {prefix}points/{prefix}weights arrays have unequal lengths ({p.shape[0]}, {w.shape[0]})")
                if np.min(w) < 0:
                    raise RuntimeError(f"{caller}: {prefix}weights array has negative value(s)")
                
            if v is not None:
                if v.ndim != 1:
                    raise RuntimeError(f"{caller}: expected {prefix}values to be a 1-d array, got shape {v.shape}")
                if p.shape[0] != v.shape[0]:
                    raise RuntimeError(f"{caller}: {prefix}points/{prefix}values arrays have unequal lengths ({p.shape[0]}, {v.shape[0]})")
                
            wtot[i] = np.sum(w) if (w is not None) else len(p)

        if np.min(wtot) <= 0:
            raise RuntimeError(f"{caller}: {prefix}weights array is all zeros")
            
        return points, weights, values, wtot
        

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
