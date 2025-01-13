from . import core
from . import utils

from .Box import Box
from .Cosmology import Cosmology
from .Catalog import Catalog

import numpy as np
from collections.abc import Iterable


class CatalogGridder:
    def __init__(self, box, cosmo, randcat, rweights, nfootprints, zcol_name='z', save_rmaps=False, save_fmaps=False, kernel='cubic'):
        r"""CatalogGridder: operates on fields $f(x)$ which are weighted sums over catalogs, and outputs Fourier-space maps.

        This is probably not the class you're looking for -- you probably want :class:`~kszx.KszPSE` instead!
        CatalogGridder is a "mid-level" class which is higher level than :func:`~kszx.estimate_power_spectrum()`,
        but is intended to b a building block for higher-level classes (:class:`~kszx.KszPSE` and perhaps
        eventually a GalaxyPSE class).

        The purpose of the CatalogGridder is to grid fields to Fourier-space maps, keeping track of
        normalizations, so that estimated power spectra have normalizations which are approximately
        correct (see below). In this docstring, we give an informal description -- for a formal
        specification, see "CatalogGridder :ref:`catalog_gridder_details`" in the sphinx docs.

        We distinguish between two types of fields which are weighted sums over catalogs:

          1. A "density" field is a weighted sum of tracers, with randoms subtracted:

             $$\delta(x) \propto \sum_i W_i \delta^3(x-x_i) - (\mbox{randoms})$$
 
          2. A "sampled" field is obtained from an underlying continuous field $F(x)$,
             by sampling at discrete points $x_i$ with sampling weights $w_i$:

             $$f(x) = \sum_i c_i \delta^3(x-x_i) \hspace{1cm} \mbox{where } c_i = w_i F(x_i) + (\mbox{noise})$$
        
        For a density field, we're interested in clustering statistics of the tracers themselves.
        For a sampled field, we're interested in clustering statsitics of the underlying continuous 
        field $F(x)$. A galaxy field $\delta_g(x)$ is a good example of a density field, and the
        kSZ velocity reconstruction $\hat v_r(x)$ is a good example of a sampled field. 

        For sampled fields, the $(c_i, w_i, F(x))$ notation deserves more explanation.
        In the following bullet points, we consider a concrete example: kSZ velocity reconstruction.

          - Recall from the overleaf that kSZ velocity reconstruction $\hat v_r$ is defined by
        
            $$\hat v_r(x) = \sum_i W_i^S \tilde T_{\rm CMB}(\theta_i) \delta^3(x-x_i)$$
        
            where $W_i^S$ is a per-galaxy weighting (e.g. FKP), and $\tilde T(\theta)$ is the 
            filtered CMB. 

          - We assume that, on large scales, the relation between $\langle \hat v_r \rangle$ 
            and $v_r^{\rm true}$ is:

            $$\langle v_r(x) \rangle = \sum_i W_i^S b_i^v v_r^{\rm true}(x_i) \delta^3(x-x_i)$$
        
          - In this example, the "underlying continuous field" $F(x)$ is the radial velocity 
            field $v_r(x)$. Comparing the last three equations, the coefficients $c_i$ and
            sampling weights $w_i$ are given by:
        
            $$c_i = W_i^S \tilde T_{CMB}(\theta_i)
            \hspace{1.5cm}
            w_i = W_i^S b_i^v$$

            We emphasize that the "sampling weight" $w_i$ is not equal to the galaxy weight $W_i^S$.
            Instead, we have $w_i = W_i^S b_i^v$, which is not initially obvious!
        
        The CatalogGridder is used as follows:

          - When a CatalogGridder is constructed, the caller specifies $N_{foot}$ "footprints". 
            Each footprint is specified by a random catalog (``randcat``) with weights (``rweights``).

          - For a density field, the constructor ``rweights`` should reflect the weighting
            that will later be applied to the galaxies (i.e. the ``gweights`` argument to 
            :meth:`~kszx.CatalogGridder.grid_density_field`). For example, if galaxies will
            be FKP-weighted, then the constructor ``rweights`` should be FKP weights for the 
            random catalog. Warning: even a small mismatch between the weighting of
            galaxies/randoms can produce a large contribution to the power spectrum!

          - For a sampled field, the constructor ``rweights`` should reflect the sampling weights $w_i$
            (defined above) that will later be applied to the galaxies. For example, in the case of
            the kSZ velocity reconstruction $\hat v_r$, we have $w_i = W_i^S b_i^v$ (not obvious,
            see above!!). Therefore, the constructor ``rweights`` should be $(W_i^S b_i^v)$,
            evaluated on the random catalog.

            (Note that for the specific case of kSZ velocity reconstruction, there is a class 
            :class:`~kszx.KszPSE` which handles these details automatically.)

          - The overall normalization of ``rweights`` is unimportant (in both the "density field"
            and "sampled field" cases).
        
          - After constructing a CatalogGridder, fields are "gridded" to Fourier space maps by
            calling :meth:`~kszx.CatalogGridder.grid_density_field` for a density field, or 
            :meth:`~kszx.CatalogGridder.grid_sampled_field` for a sampled field.

            When fields are gridded, a ``footprint_index`` is specified, to provide the appropriate
            normalization. In the case of a density field, random subtraction is also performed.
        
            WARNING: Fourier-space maps returned by these methods have an unusual normalization,
            which is chosen to give sensible power spectrum estimates (see next bullet point), but
            may not be what you expect! For more info, see "CatalogGridder :ref:`catalog_gridder_details`" 
            in the sphinx docs.

          - After making Fourier-space maps, you compute "unnormalized" power spectra by calling
            the low-level function :func:`~kszx.estimate_power_spectrum`, and then normalize
            by multiplying by ``catalog_gridder.ps_normalization[i,j]``, where $i$ and $j$ are
            the footprint indices.

            The ``ps_normalization`` member is computed in the constructor, and is a matrix $N_{ij}$
            which can be used to normalize the power spectrum for gridded fields on footprints $i,j$.
            This normalization is an ansatz which should be approximately correct, but may break down
            on large scales. (Eventually, we'll implement rigorous window function deconvolution.)

        When constructor arguments are described in the bullet points below, we sometimes use pluralized
        notation. For example, the ``randcat`` arg has type kszx.Catalog(s). This means that the ``randcat``
        arg can either be a length-``nfootprints`` list of Catalog objects, or a single Catalog object.
        (Specifying a single Catalog is equivalent to repeating the Catalog ``nfootprints`` times.)
        
        Constructor arguments:

           - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
             See :class:`~kszx.Box` for more info.

           - ``cosmo`` (kszx.Cosmology): see :class:`~kszx.Cosmology`.
             Only used to convert redshifts to distances.
        
           - ``randcat`` (kszx.Catalog(s)): see next item.

           - ``rweights`` (scalar(s) or 1-d arrays): as described above, the CatalogGridder
             contains $N_{foot}$ footprints. Each footprint is specified by a random catalog
             (``randcat``) with weights (``rweights``).

             The ``rweights`` for each footprint can either be a 1-d array of length ``randcat.size``,
             or a scalar (if every object has the same weight). See above for more discussion of
             how to choose ``rweights``.
        
           - ``nfootprints`` (int): Number of footprints $N$.
        
           - ``zcol_name`` (string(s)): Each ``randcat`` must contain columns ``'ra_deg'``,
             ``'dec_deg'``, and a redshift column which is ``'z'`` by default, but this can
             be overridden by specifying the ``zcol_name`` argument.

             For example, in a photometric randcat, you might have two columns ``'ztrue'``
             and ``'zobs'``. In this case, you probably want ``zcol_name = 'zobs'``, in
             order to grid catalogs at observed redshifts, not true (unobserved) redshifts.
        
           - ``save_rmaps`` (boolean(s)): If ``True`` for a footprint, then a real-space gridded
             footprint will be saved. **This is needed in order to call** :meth:`grid_density_field`
             **on the footprint.** It consumes significant memory, so you may want to set ``save_rmaps``
             to True only on footprints where ``grid_density_field()`` will be called.

             (If you call ``grid_density_field()`` but forgot to set ``save_rmaps=True``, then
             you'll get a verbose error message.)
        
           - ``save_fmaps`` (boolean(s)): If ``True`` for a footprint, then a Fourier-space
             gridded footprint will be saved. This is only useful for debugging.
        
           - ``kernel`` (string): either ``'cic'`` or ``'cubic'`` (more options will be defined later).

        Class members:

           - ``ps_normalization`` (matrix): An $N_{\rm foot}$-by-$N_{\rm foot}$ matrix $N_{ij}$.

             Multiplying by $N_{ij}$ is the final step in computing normalized power spectrum
             estimates. (After computing Fourier-space maps with :meth:`grid_density_field()`
             and/or :meth:`grid_sampled_field()`, then computing unnormalized power spectra
             the low-level function :func:`~kszx.estimate_power_spectrum()`.)
        """
        
        assert isinstance(box, Box)
        assert isinstance(cosmo, Cosmology)
        assert kernel in ['cic','cubic']  # more kernels coming soon
        assert int(nfootprints) == nfootprints
        assert nfootprints >= 1
        assert nfootprints <= 50

        self.box = box
        self.cosmo = cosmo
        self.nfootprints = int(nfootprints)
        self.kernel = kernel
        
        # "Vector" args:
        #   randcats     length-nfootprints list of Catalogs
        #   zcol_names   length-nfootprints list of strings
        #   save_rmaps   length-nfootprints list of booleans
        #   save_fmaps   length-nfootprints list of booleans
        #   rweights     length-nfootprints list of (0-d or 1-d numpy array)

        self.randcats = self._parse_vector_arg(randcat, 'randcat', Catalog)
        self.zcol_names = self._parse_vector_arg(zcol_name, 'zcol_name', str)
        self.save_rmaps = self._parse_vector_arg(save_rmaps, 'save_rmaps', bool)
        self.save_fmaps = self._parse_vector_arg(save_fmaps, 'save_fmaps', bool)
        self.rweights = self._parse_rweights(rweights)

        assert all(rcat.size >= 1000 for rcat in self.randcats)
        
        self.rmaps = [ None for _ in range(self.nfootprints) ]  # real-space maps
        self.fmaps = [ None for _ in range(self.nfootprints) ]  # Fourier space maps
        self.rwsum = np.zeros(self.nfootprints)

        # FIXME this loop can be optimized, to avoid repeating computations in cases
        # where different fields use the same (randcat, rweights, zcol_name).
        
        for i in range(self.nfootprints):
            rc = self.randcats[i]
            rw = self.rweights[i]
            points = rc.get_xyz(cosmo, zcol_name = self.zcol_names[i])
            
            assert (rw.ndim == 0) or (rw.shape == (rc.size,))
            rw = rw if (rw.ndim > 0) else np.full(rc.size, rw)
                
            self.rwsum[i] = np.sum(rw)
            self.rmaps[i] = core.grid_points(box, points, rw, kernel=kernel)
            self.fmaps[i] = core.fft_r2c(box, self.rmaps[i])

            assert self.rwsum[i] != 0
            core.apply_kernel_compensation(box, self.fmaps[i], kernel)

            if not self.save_rmaps[i]:
                self.rmaps[i] = None

        # FIXME revisit the issue of choosing K!
        K = 0.6 * box.knyq
        self.A = self.compute_A(K)
        assert self.A.shape == (self.nfootprints, self.nfootprints)
        assert np.all(self.A.diagonal() > 0)

        # Clear fmaps after calling self.compute_A().
        for i in range(self.nfootprints):
            if not self.save_fmaps[i]:
                self.fmaps[i] = None
            
        self.ps_normalization = np.zeros((self.nfootprints, self.nfootprints))
        
        for i in range(self.nfootprints):
            for j in range(i+1):
                r = self.A[i,j] / np.sqrt(self.A[i,i] * self.A[j,j])
                if np.abs(r) < 0.03:
                    print(f'CatalogGridder: fields {(j,i)} have non-overlapping footprints, cross power will not be estimated')
                else:
                    self.ps_normalization[i,j] = self.ps_normalization[j,i] = 1.0 / self.A[i,j]


    def grid_density_field(self, gcat, gweights, footprint_index, zcol_name='z'):
        r"""Grids a density field, subtracts randoms (with footprint_index), and returns a Fourier-space map.

        The density field is specified by points $x_i$ (via the ``gcat`` and ``zcol_name`` args),
        and weights $W_i$ (via the ``gweights`` arg), and is defined by:

        $$\delta(x) \propto \sum_i W_i \delta^3(x-x_i) - \big( \mbox{randoms} \big)$$

        where we have swept the normalization under the rug by using the $\propto$ symbol.
        The grid_density_field() method will assign a normalization which is appropriate if you're
        interested in clustering statistics of the *overdensity* field $\delta = \rho/\bar\rho$.
        For a precise specification of the normalization, see 
        "CatalogGridder :ref:`catalog_gridder_details`" in the sphinx docs.
        
        Function arguments:

          - ``gcat`` (:class:`Catalog`): Defines spatial locations $x_i$.

          - ``gweights`` (scalar, or 1-d numpy array of length ``gcat.size``):
            Defines weights $W_i$.
        
          - ``footprint_index`` (integer): An integer $0 \le i < N_{\rm footprints}$.
            Specifies which "footprint" (see constructor docstring) is used to subtract
            randoms, and for normalization.

          - ``zcol_name``` (string):  The catalog ``gcat`` must contain columns ``'ra_deg'``,
            ``'dec_deg'``, and a redshift column which is ``'z'`` by default, but this can
            be overridden by specifying the ``zcol_name`` argument.

            For example, in a photometric catalog, you might have two columns ``'ztrue'``
            and ``'zobs'``. In this case, you probably want ``zcol_name = 'zobs'``, in
            order to put tracers at their observed redshifts, not true (unobserved) redshifts.
        """
        
        assert isinstance(gcat, Catalog)
        assert 0 <= footprint_index < self.nfootprints

        if not self.save_rmaps[footprint_index]:
            raise RuntimeError(f"CatalogGridder.grid_density_field() was called with {footprint_index=},"
                               + f" but save_rmaps=False for this footprint. You probably want to set"
                               + f" save_rmaps=True in the constructor, but only for footprint indices"
                               + f" where grid_density_field() will be called. See docstring for more info.")
        
        rwsum = self.rwsum[footprint_index]
        points = gcat.get_xyz(self.cosmo, zcol_name=zcol_name)
        gweights = self._parse_coeffs(gcat, gweights, 'CatalogGridder.grid_density_field()', 'gweights')

        wsum = np.sum(gweights)
        assert wsum != 0.0

        rmap = core.grid_points(self.box, points, (rwsum/wsum) * gweights, kernel=self.kernel)
        rmap -= self.rmaps[footprint_index]   # subtract randoms

        fmap = core.fft_r2c(self.box, rmap)
        core.apply_kernel_compensation(self.box, fmap, self.kernel)
        return fmap


    def grid_sampled_field(self, gcat, coeffs, wsum, footprint_index, spin=0, zcol_name='z'):
        r"""Grids a sampled field $f(x) = \sum_i c_i \delta^3(x-x_i)$, and returns a Fourier-space map.

        A "sampled" field is obtained from an underlying continuous field $F(x)$, by sampling
        at discrete points $x_i$ with weights $w_i$:

        $$f(x) = \sum_i c_i \delta^3(x-x_i) \hspace{1cm} \mbox{where } c_i = w_i F(x_i) + (\mbox{noise})$$

        For example, kSZ velocity reconstruction (summarizing discussion from the class 
        :class:`~kszx.CatalogGridder` docstring):

        $$\begin{align}
        \hat v_r(x) &= \sum_i W_i^S \tilde T_{\rm CMB}(\theta_i) \delta^3(x-x_i) \\
        c_i &= W_i^S \tilde T_{CMB}(\theta_i) \\
        w_i &= W_i^S b_i^v
        \end{align}$$

        (Note that for the specific case of kSZ velocity reconstruction, there is a higher-level class
        :class:`~kszx.KszPSE`, and you shouldn't need to use :class:`~kszx.CatlogGridder` directly.)

        The grid_sampled_field() method will assign a normalization which is appropriate if you're
        interested in clustering statistics of the *underlying continuous* field $F(x)$.
        For a precise specification of the normalization, see 
        "CatalogGridder :ref:`catalog_gridder_details`" in the sphinx docs.

        Function arguments:

          - ``gcat`` (:class:`Catalog`): Defines spatial locations $x_i$.

          - ``coeffs`` (scalar, or 1-d numpy array of length ``gcat.size``):
            Coefficients $c_i$ appearing in the sampled field $f(x) = \sum_i c_i \delta^3(x-x_i)$.

          - ``wsum`` (scalar): The sum of the sampling weights $\sum_i w_i$.
            This is needed to assign a normalization to the output Fourier-space map.
            (Note that we don't need the individual sampling weights $w_i$, just their sum.)

          - ``footprint_index`` (integer): An integer $0 \le i < N_{\rm footprints}$.
            Specifies which "footprint" (see constructor docstring) is used to compute
            the normalization.

          - ``spin`` (integer): The "spin" of the FFT called at the end of grid_sampled_field()
            to go from real space to Fourier space. You should set ``spin=0`` for a scalar field
            (e.g. surrogate galaxy field $S_g(x)$), and ``spin=1`` for a radial velocity (e.g.
            kSZ velocity reconstruction $\hat v_r(x)$. For the precise definition of ``spin``,
            see FFT :ref:`fft_conventions` in the sphinx docs.

          - ``zcol_name`` (string):  The catalog ``gcat`` must contain columns ``'ra_deg'``,
            ``'dec_deg'``, and a redshift column which is ``'z'`` by default, but this can
            be overridden by specifying the ``zcol_name`` argument.

            For example, in a photometric catalog, you might have two columns ``'ztrue'``
            and ``'zobs'``. In this case, you probably want ``zcol_name = 'zobs'``, in
            order to put tracers at their observed redshifts, not true (unobserved) redshifts.
        """
        
        assert isinstance(gcat, Catalog)
        assert 0 <= footprint_index < self.nfootprints
        assert wsum != 0.0

        rwsum = self.rwsum[footprint_index]
        points = gcat.get_xyz(self.cosmo, zcol_name=zcol_name)
        coeffs = self._parse_coeffs(gcat, coeffs, 'CatalogGridder.grid_sampled_field()', 'coeffs')

        fmap = core.grid_points(self.box, points, (rwsum/wsum) * coeffs, kernel=self.kernel, fft=True, spin=spin)
        core.apply_kernel_compensation(self.box, fmap, self.kernel)
        return fmap
        

    def compute_A(self, K):
        r"""Computes the $A$-matrix (an intermediate quantity used to compute PS normalization). Intended for debugging!

        You shouldn't need to call this function, unless you're interested in the details of the power spectrum
        normalization! For a definition of what it computes, see "CatalogGridder :ref:`catalog_gridder_details`" 
        in the sphinx docs, where the $A$-matrix is denoted $A_{RR'}$.

        Returns a shape (nfootprints, nfootprints) array. In order to call ``compute_A()``, you need to call 
        the ``CatalogGridder`` constructor with ``save_fmaps=True``."""

        if any(x is None for x in self.fmaps):
            raise RuntimeError('CatalogGridder: in order to call compute_A(), you should call'
                               + ' the CatalogGridder constructor with save_fmaps=True')
            
        edges = np.array([ 0, K / 2**(1./self.box.ndim), K ])

        # use_dc=True is important here!!
        pk, counts = core.estimate_power_spectrum(self.box, self.fmaps, edges, use_dc=True, return_counts=True)
        assert pk.shape == (self.nfootprints, self.nfootprints, 2)
        assert counts.shape == (2,)

        # Note factor (1/V_box), which normalizes sum_k -> int d^3k/(2pi)^3
        return (pk[:,:,0] - pk[:,:,1]) * counts[0] / self.box.box_volume


    ####################################################################################################
    

    def _parse_vector_arg(self, arg, arg_name, _type):
        if isinstance(arg, _type):
            return [ arg ] * self.nfootprints

        try:
            n = len(arg)
        except:
            raise RuntimeError(f'CatalogGridder: expected {arg_name} to be either {type}, or list of {type}')

        if n != self.nfootprints:
            raise RuntimeError(f'CatalogGridder: expected {arg_name} to have length nfootprints={self.nfootprints}, got length {n}')

        arg = list(arg)
        assert len(arg) == n

        for i in range(n):
            if not isinstance(arg[i], _type):
                raise RuntimeError(f'CatalogGridder: expected {arg_name} to be either {type}, or list of {type}')                

        return arg


    def _parse_rweights(self, rweights):
        if self._is_scalar(rweights):
            rweights = np.array(float(rweights))   # 0-d numpy array
            return [ rweights ] * self.nfootprints

        if isinstance(rweights,np.ndarray) and all(rweights.shape == (rcat.size,) for rcat in self.randcats):
            rweights = utils.asarray(rweights, 'CatalogGridder', 'rweights', dtype=float)
            return [ rweights ] * self.nfootprints

        try:
            n = len(rweights)
        except:
            raise RuntimeError("CatalogGridder: couldn't parse 'rweights' (neither iterable nor scalar)")

        if n != self.nfootprints:
            raise RuntimeError(f"CatalogGridder: couldn't parse 'rweights' (got length {n}, expected"
                               + f" length nfootprints={self.nfootprints})")
        
        rweights = [ utils.asarray(w, 'CatalogGridder', 'rweights', dtype=float) for w in rweights ]
        assert len(rweights) == self.nfootprints

        for i in range(self.nfootprints):
            if (rweights[i].ndim > 0) and (rweights[i].shape != (self.randcats[i].size,)):
                raise RuntimeError(f"CatalogGridder: expected rweights[{i}] to be a scalar or length"
                                   + f" randcats[{i}].size={self.randcats[i].size}, got shape" 
                                   + f" {rweights[i].shape}")

        return rweights
    
            
    @staticmethod
    def _is_scalar(f):
        try:
            f = float(f)
            return True
        except:
            pass
        return False


    def _parse_coeffs(self, gcat, coeffs, func_name, arg_name):
        """Helper for grid_density_field() and grid_sampled_field()."""
        
        coeffs = utils.asarray(coeffs, func_name, arg_name, dtype=float)
        
        if coeffs.shape == (gcat.size,):
            return coeffs
        if coeffs.ndim == 0:
            return np.full(gcat.size, coeffs)
        
        raise RuntimeError(f"{func_name}: expected {arg_name} to have length {gcat.size}"
                           + f" (=gcat.size), got shape {coeffs.shape}")
