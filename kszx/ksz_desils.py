"""This source file contains some code from Selim's DESILS KSZ notebooks."""

import os
import time
import yaml
import shutil
import functools
import numpy as np

from . import io_utils
from . import utils

from .Catalog import Catalog
from .Cosmology import Cosmology
from .KszPSE import KszPSE

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def subtract_zbin_means(w, z, nz=15):
    r"""Given per-galaxy weights 'w', and galaxy redshifts 'z', compute and subtract the mean weight in redshift bins.
    
    Function arguments:
      - ``w`` (array): per-galaxy weights
      - ``z`` (array): galaxy redshifts

    Return value:
      - A copy of ``w``, after subtracting the mean weight in redshift bins.

    (Source: Selim's ``pipeline_getsurrogates_selim.ipynb``, Nov 2024.)
    """

    w = np.asarray(w)
    z = np.asarray(z)
    assert w.shape == z.shape
    
    zmin = np.min(z) - 1.0e-10
    zmax = np.max(z) + 1.0e-10
    zbins = np.linspace(zmin, zmax, nz)
    
    locatez = np.digitize(z, zbins)
    zbin_means = np.zeros(nz)
    
    for iz in range(nz):
        wbin = w[locatez==iz]
        if len(wbin) > 0:
            zbin_means[iz] = np.mean(wbin)
    
    return w - zbin_means[locatez]


class PhotozDistribution:
    def __init__(self, zobs_arr, zerr_arr, zmin=0.0, zmax=1.5, zerr_min=0.02, zerr_max=0.5, nzbins=100, nzerrbins=49, niter=100, sigma=2):
        r"""Selim's model for the joint (ztrue, zobs, zerr) distribution in DESILS-LRG.

        Obtained by Richardson-Lucy deconvolution on the observed (zobs, zerr) distribution.
        (Source: Selim's ``pipeline_getsurrogates_selim.ipynb``, Nov 2024.)

        Constructor args:

          - ``zobs_arr``: 1-d array of observed redshifts.
          - ``zerr_arr``: 1-d array of observed redshift errors.
          - ``zmin, zmax, nzbins``: used internally when doing Richardson-Lucy deconvolution. 
            The defaults (0, 1.5, 100) are appropriate for DESILS-LRG.
          - ``zerr_min, zerr_max, nzerrbins``: used internally to split up the data before doing Richardson-Lucy deconvolution.
            The defaults (0.02, 0.5, 49) are approriate for DESILS-LRG.
          - ``niter``: number of Richardson-Lucy iterations.
          - ``sigma``: adjust for different smoothing levels.
        """
        
        zobs_arr = np.asarray(zobs_arr)
        zerr_arr = np.asarray(zerr_arr)
        
        # Argument checking.
        assert zobs_arr.shape == zerr_arr.shape
        assert zobs_arr.ndim == 1
        assert 0 <= zmin < zmax
        assert 0 <= zerr_min < zerr_max
        assert nzbins > 0
        assert nzerrbins > 0
        assert niter > 0
        assert sigma > 0.0

        self.nzbins = nzbins
        self.nzerrbins = nzerrbins
        self.z = np.linspace(zmin, zmax, nzbins)
        self.pzebins = np.linspace(zerr_min, zerr_max, nzerrbins+1)
        self.meanphotz = np.zeros(nzerrbins+2)
        self.umat = np.zeros((nzerrbins+2, nzbins))
        
        locatepze = np.digitize(zerr_arr, self.pzebins)

        for iz in range(nzerrbins+2):
            ix_bin = (locatepze == iz)
            zobs_bin = zobs_arr[ix_bin]
            zerr_bin = zerr_arr[ix_bin]
    
            if len(zerr_bin) < 5:
                continue
    
            self.meanphotz[iz] = np.mean(zerr_bin)
            counts_obs, bin_edges = np.histogram(zobs_bin, density=False, bins=nzbins, range=(zmin,zmax))
    
            zs_obs = (bin_edges[1:]+bin_edges[:-1])/2

            counts_obs = gaussian_filter1d(counts_obs, sigma=sigma)*1.
            counts_obs /= np.sum(counts_obs)
    
            counts_obs_int = interp1d(zs_obs, counts_obs, fill_value=0, bounds_error=False)
            observed_dist = counts_obs_int(self.z)

            Pij = np.array([[np.exp(-0.5*((zi-zj)/self.meanphotz[iz])**2) for zi in self.z] for zj in self.z])
            for j in range(len(self.z)):
                Pij[:,j] /= np.sum(Pij[:,j])

            di = observed_dist
            uguess = np.ones(nzbins)
    
            for iteration in range(niter):
                ci = np.dot(Pij, uguess)
                uguess *= np.dot(di / (ci+1.0e-10), Pij)

            self.umat[iz,:] = uguess / np.sum(uguess)

    
    def sample_z(self, zerr, zmin=0.4, zmax=1.0):
        r"""Randomly sample (ztrue, zobs) pairs, conditioned on a specified array of 'zerr' values.

        Function arguments:

          - ``zerr`` (array): 1-d array of zerr values.
          - ``(zmin, zmax)``: the returned zobs values will be constrained to satisfy 
            $z_{\rm min} < z_{\rm obs} < z_{\rm max}$. The defaults (0.4, 1.0) are appropriate
            for DESILS-LRG.

        Return values:

          - ``ztrue``: 1-d array (same length as ``zerr`` argument).
          - ``zobs``: 1-d array (same length as ``zerr`` argument).

        Each returned $(z_{\rm true}, z_{\rm obs})$ pair is consistent with the corresponding
        caller-specified $z_{\rm err}$.
        """
        
        zerr = np.asarray(zerr)
        assert zerr.ndim == 1

        locatepze = np.digitize(zerr, self.pzebins)
        ztrue = np.zeros_like(zerr)
        zobs = np.zeros_like(zerr)

        for iz in range(self.nzerrbins + 2):
            ix_bin = (locatepze == iz)
            lentruez = np.sum(ix_bin)
            
            buffer = 4
            lenpze = lentruez * buffer
    
            # Define some oversampled z-bins.
            # KMS: changed number of bins from lenpze to (10 * self.nzbins)
            # (speeds things up and gives similar results)
            
            zbin_edges = np.linspace(self.z[0], self.z[-1], 10*self.nzbins + 1)
            dz = zbin_edges[1] - zbin_edges[0]

            interp = interp1d(self.z, self.umat[iz,:])
            zvecs = (zbin_edges[1:] + zbin_edges[:-1]) / 2.  # bin centers
            zintrp = np.maximum(interp(zvecs), 0)

            # We perturb true_z within its bin [-dz/2,dz/2]
            # (Otherwise, all true_z's are at bin centers, and the histogram looks weird)
            
            true_z_sample = np.random.choice(zvecs, size=lenpze, p = (zintrp / np.sum(zintrp)))
            true_z_sample += np.random.uniform(-dz/2, dz/2, size=lenpze)
            photoz_err = np.random.normal(scale = self.meanphotz[iz], size = lenpze)
            observed_z_sample = true_z_sample + photoz_err 

            selectz_ = (observed_z_sample > zmin) * (observed_z_sample < zmax)
            ztrue[ix_bin] = (true_z_sample[selectz_])[:lentruez]
            zobs[ix_bin] = (observed_z_sample[selectz_])[:lentruez]

        return ztrue, zobs


####################################################################################################


class Kpipe:
    def __init__(self, input_dir, output_dir, nsurr=400):
        """Kendrick KSZ P(k) pipeline.

        Runnable either by calling the run() method, or from the command line with:
           python -m kszx <input_dir> <output_dir>

        Reminder: input directory must contain files { params.yml, galaxies.h5, randoms.h5,
        bounding_box.pkl }. The pipeline will populate the output directory with files
        { params.yml, pk_data.npy, pk_surrogates.npy }. (For details of these file formats,
        see one of the README files that Kendrick or Selim have lying around.)

        Note that 'class Kpipe', 'class KszPSE', and 'class CatalogGridder' are related
        as follows:

          - Kpipe: high-level pipeline, knows about file inputs/outputs, defers
              heavy lifting to 'class KszPSE'.

          - KszPSE: KSZ power spectrum estimation and surrogate generation logic
              is here.

          - CatalogGridder: a lower-level class which supplies normalizations (both
              field-level and power spectrum level).
        """
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.nsurr = nsurr
        
        # Create output_dir and 'tmp' subdir
        os.makedirs(f'{output_dir}/tmp', exist_ok=True)

        with open(f'{input_dir}/params.yml', 'r') as f:
            params = yaml.safe_load(f)

            self.surr_bg = params['surr_bg']
            self.nkbins = params['nkbins']
            self.kmax = params['kmax']
            self.kbin_edges = np.linspace(0, self.kmax, self.nkbins+1)
            self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.

        self.box = io_utils.read_pickle(f'{input_dir}/bounding_box.pkl')

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
    def pse(self):
        """Returns a KszPSE object."""
        
        print('Initializing KszPSE: this will take a few minutes')
        
        # FIXME needs comment
        surr_ngal_mean = self.gcat.size
        surr_ngal_rms = 4 * np.sqrt(self.gcat.size)  # 4x Poisson

        pse = KszPSE(
            box = self.box, 
            cosmo = self.cosmo, 
            randcat = self.rcat, 
            kbin_edges = self.kbin_edges,
            surr_ngal_mean = surr_ngal_mean,
            surr_ngal_rms = surr_ngal_rms,
            surr_bg = self.surr_bg,
            rweights = self.rcat.weight_zerr,
            nksz = 2,
            # ksz_rweights = None,
            ksz_bv = [ self.rcat.bv_90, self.rcat.bv_150 ],
            ksz_tcmb_realization = [ self.rcat.tcmb_90, self.rcat.tcmb_150 ],
            ztrue_col = 'ztrue',
            zobs_col = 'zobs'
        )
        
        print('KszPSE initialization done')
        return pse

    
    def get_pk_data(self, run=False):
        """Returns a shape (3,3,nkbins) array.

        If run=False, then this function expects the P(k) file to be on disk from a previous pipeline run.
        If run=True, then the P(k) file will be computed if it is not on disk.
        """

        if os.path.exists(self.pk_data_filename):
            return io_utils.read_npy(self.pk_data_filename)
        
        if not run:
            raise RuntimeError(f'Kpipe.get_pk_data(): run=False was specified, and file {self.pk_data_filename} not found')

        t90 = subtract_zbin_means(self.gcat.tcmb_90, self.gcat.z, nz=25)
        t150 = subtract_zbin_means(self.gcat.tcmb_150, self.gcat.z, nz=25)

        pk_data = self.pse.eval_pk(
            gcat = self.gcat,
            gweights = self.gcat.weight_zerr,
            # ksz_gweights = None, 
            ksz_bv = [ self.gcat.bv_90, self.gcat.bv_150 ], 
            ksz_tcmb = [ t90, t150 ],
            zobs_col = 'z'
        )

        io_utils.write_npy(self.pk_data_filename, pk_data)
        return pk_data


    def get_pk_surrogate(self, isurr, run=False):
        """Returns a shape (6,6,nkbins) array.
        
        If run=False, then this function expects the P(k) file to be on disk from a previous pipeline run.
        If run=True, then the P(k) file will be computed if it is not on disk.
        """
        
        fname = self.pk_single_surr_filenames[isurr]
        
        if os.path.exists(fname):
            return io_utils.read_npy(fname)

        if not run:
            raise RuntimeError(f'Kpipe.get_pk_surrogate(): run=False was specified, and file {fname} not found')

        self.pse.simulate_surrogate()
        
        for sv in [ self.pse.Sv_noise, self.pse.Sv_signal ]:
            assert sv.shape ==  (2, self.rcat.size)
            for j in range(2):
                sv[j,:] = subtract_zbin_means(sv[j,:], self.rcat.zobs, nz=25)
    
        pk = self.pse.eval_pk_surrogate()

        io_utils.write_npy(fname, pk)
        return pk


    def get_pk_surrogates(self):
        """Returns a shape (nsurr,6,6,nkins) array, containing P(k) for all surrogates."""
        
        if os.path.exists(self.pk_surr_filename):
            return io_utils.read_npy(self.pk_surr_filename)

        if not all(os.path.exists(f) for f in self.pk_single_surr_filenames):
            raise RuntimeError(f'Kpipe.read_pk_surrogates(): necessary files do not exist; you need to call Kpipe.run()')

        pk = np.array([ io_utils.read_npy(f) for f in self.pk_single_surr_filenames ])
        
        io_utils.write_npy(self.pk_surr_filename, pk)
        return pk


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
            print(f'Kpipe.run(): pipeline has already been run, exiting early')
            return
        
        # Initialize KszPSE before creating multiprocessing Pool.
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


####################################################################################################


class RegulatedDeconvolver:
    def __init__(self, zobs_vec, zerr_vec, zbin_width, soft_zmax=None):
        """A photo-z error deconvolver which avoids the oscillatory behavior of Lucy-Richardson.
        
        Given the observed 2-d (zobs,zerr) distribution, models the 3-d (ztrue,zobs,zerr) distribution.
        FIXME: needs comments explaining how it works!
        
        - zobs_vec: array of shape (ngal,) containing observed (photometric) redshifts.

        - zerr_vec: array of shape (ngal,) containing estimated photo-z errors.

        - zbin_width (scalar): used internally for binning. (Recommend ~0.01 for DESILS-LRG.)

        - soft_zmax (scalar): used internally; galaxies with (zobs > soft_zmax) are discarded.
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
        """Helper for sample(): Sample from conditional distribution P(ztrue|zobs,zerr)."""
        
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
        """Returns (ztrue, zobs, zerr), where all 3 arrays are shape (n,).'
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
        """Returns (ztrue, zobs, zerr), where all 3 arrays are shape (n,).
        
        Uses a multiprocessing Pool for speed. If 'processes' is None, then a
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
