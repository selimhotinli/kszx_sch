import numpy as np
import scipy.special

from . import utils

# FIXME some day I'll write comments explaining how the RegulatedDeconvolver actually works.
# In the meantime I apologize for this cryptic code!


class RegulatedDeconvolver:
    def __init__(self, zobs_vec, zerr_vec, zbin_width, soft_zmax=None):
        r"""A photo-z error deconvolver which avoids the oscillatory behavior of Lucy-Richardson.

        The constructor takes a sequence of samples from the observed 2-d $(z_{obs}, z_{err})$
        distribution, and builds a model for the 3-d $(z_{true}, z_{obs}, z_{err})$ distribution.
        
        This 3-d distribution can be sampled with :meth:`~kszx.RegulatedDeconvolver.sample()`
        or :meth:`~kszx.RegulatedDeconvolver.sample_parallel()`. This is used in our DESILS-LRG
        pipeline, to generate triples $(z_{true}, z_{obs}, z_{err})$ for the random catalog,
        given pairs $(z_{obs}, z_{err})$ from the galaxy catalog.
        
        - ``zobs_vec``: 1-d array containing observed (photometric) redshifts.

        - ``zerr_vec``: 1-d array containing estimated photo-z errors.

        - ``zbin_width`` (scalar): used internally for binning. (Recommend ~0.01 for DESILS-LRG.)

        - ``soft_zmax`` (scalar): used internally; galaxies with (zobs > soft_zmax) are discarded.
          Should be chosen significantly larger than the max redshift of interest.
          (For DESILS-LRG with zmax=1, recommend soft_zmax=1.5. If this parameter is omitted,
          then you'll get a tail of rare galaxies with z~5, which is annoying.)
        """
        
        assert zobs_vec.ndim == 1
        assert zobs_vec.shape == zerr_vec.shape
        assert np.all(zobs_vec >= 0)
        assert np.all(zerr_vec > 0)

        if soft_zmax is not None:
            mask = (zobs_vec <= soft_zmax)
            zobs_vec = zobs_vec[mask]
            zerr_vec = zerr_vec[mask]
        
        self.zobs_vec = zobs_vec
        self.zerr_vec = zerr_vec
        self.zbin_width = zbin_width
        self.zmin = np.min(zobs_vec)
        self.zmax = np.max(zobs_vec)
        self.soft_zmax = soft_zmax

        self.nzbins = np.array(self.zmax/zbin_width, dtype=int)
        self.zbin_edges = np.arange(self.nzbins+1) * zbin_width
        
        # The variable y = log(zerr + zbin_width) is used to define zerr bins.
        y = np.log(zerr_vec + self.zbin_width)
        self.ymin = np.min(y) - 1.0e-8
        self.ymax = np.max(y) + 1.0e-8
        self.nybins = int((self.ymax - self.ymin) / 0.2) + 1
        self.ybin_width = (self.ymax - self.ymin) / self.nybins
        
        iz = np.array(zobs_vec / zbin_width, dtype=int)
        iz = np.clip(iz, 0, self.nzbins-1)

        iy = np.array((y - self.ymin) / self.ybin_width, dtype=int)
        iy = np.clip(iy, 0, self.nybins-1)

        # self.pobs[iy,iz] is an array of shape (self.nybins, self.nzbins).
        self.pobs = np.bincount(iy*self.nzbins + iz, minlength=self.nybins*self.nzbins)
        self.pobs = np.reshape(self.pobs, (self.nybins, self.nzbins))
        self.pobs += 1   # regulate
        self.pobs = self.pobs / np.sum(self.pobs,axis=1).reshape((-1,1))  # normalize to PDF in z-bins (for each y-bin)


    def _sample(self, zobs, zerr):
        r"""Helper for sample(): Sample from conditional distribution $P(z_{true} | z_{obs},z_{err})$."""
        
        n = len(zobs)
        nb = self.nzbins
        rng = np.random.default_rng()

        # iy, iz = bin indices (shape (n,))
        # Reminder: the variable y = log(zerr + zbin_width) is used to define zerr bins.
        y = np.log(zerr + self.zbin_width)
        iy = np.array((y - self.ymin) / self.ybin_width, dtype=int)
        iy = np.clip(iy, 0, self.nybins-1)
        
        # pobs = shape(n,nb)
        pobs = self.pobs[iy]
        assert pobs.shape == (n,nb)

        # cdf = shape (n,nb+1), not "reweighted by pobs".
        cdf = self.C((self.zbin_edges.reshape((1,-1)) - zobs.reshape((-1,1))) / zerr.reshape((-1,1)))
        assert cdf.shape == (n,nb+1)
        
        # pdf = shape (n,nb), "reweighted by pobs"
        pdf = cdf[:,1:] - cdf[:,:-1]
        pdf *= pobs
        pdf /= np.sum(pdf,axis=1).reshape((-1,1))

        # ztrue_bins has shape (n,). I don't think there's any way to vectorize this loop.
        ztrue_bins = np.array([ rng.choice(nb, p=pdf[i,:]) for i in range(n) ])
        
        cdf_lo = cdf[np.arange(n), ztrue_bins]
        cdf_hi = cdf[np.arange(n), ztrue_bins+1]
        c = cdf_lo + (cdf_hi-cdf_lo) * rng.uniform(size=n)
        ztrue = zobs + zerr * self.Cinv(c)

        return ztrue

        
    def sample(self, n, zobs_min=None, zobs_max=None):
        r"""Returns ``(ztrue, zobs, zerr)``, where all 3 arrays have shape ``(n,)``.
        
        If n is large, you may want to call sample_parallel() instead."""

        zobs_min = zobs_min if (zobs_min is not None) else (self.zmin - 0.01)
        zobs_max = zobs_max if (zobs_max is not None) else (self.zmax + 0.01)
        mask = np.logical_and(self.zobs_vec >= zobs_min, self.zobs_vec <= zobs_max)
        rng = np.random.default_rng()
        
        # Shape (n,)
        ix = np.nonzero(mask)[0]
        ix = rng.choice(ix, size=n, replace=True)
        zobs = self.zobs_vec[ix]
        zerr = self.zerr_vec[ix]
        ztrue = np.zeros(n)
        pos = 0

        # Compute in batches, to keep memory usage under control.
        while pos < n:
            pos2 = min(n, pos + 10**5)
            ztrue[pos:pos2] = self._sample(zobs[pos:pos2], zerr[pos:pos2])
            pos = pos2
                
        return ztrue, zobs, zerr


    def sample_parallel(self, n, zobs_min=None, zobs_max=None, processes=None):
        r"""Returns ``(ztrue, zobs, zerr)``, where all 3 arrays are shape ``(n,)``.
        
        Uses a multiprocessing Pool for speed. If ``processes`` is None, then a
        sensible default will be chosen."""

        with utils.Pool(processes) as pool:
            p = pool._processes
            nlist = [ (((i+1)*n)//p) - ((i*n)//p) for i in range(p) ]
            m = pool.starmap_async(self.sample, [(n,zobs_min,zobs_max) for n in nlist])
            result = m.get()  # list of (ztrue, zobs, zerr) triples
            
        # Concatenate results from each task.
        ztrue = np.concatenate([r[0] for r in result])
        zobs = np.concatenate([r[1] for r in result])
        zerr = np.concatenate([r[2] for r in result])

        return ztrue, zobs, zerr


    @classmethod
    def C(cls, x):
        return (1/2) * scipy.special.erf(x / np.sqrt(2.))

    @classmethod
    def Cinv(cls, y):
        return scipy.special.erfinv(2*y) * np.sqrt(2.)
