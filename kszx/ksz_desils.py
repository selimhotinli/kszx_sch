"""This source file contains some code from Selim's DESILS KSZ notebooks."""

import os
import time
import yaml
import shutil
import functools
import numpy as np
import scipy.linalg
import scipy.special
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from . import core
from . import utils
from . import io_utils
from . import wfunc_utils

from .Catalog import Catalog
from .Cosmology import Cosmology
from .KszPSE import KszPSE
from .SurrogateFactory import SurrogateFactory


####################################################################################################


class PhotozDistribution:
    def __init__(self, zobs_arr, zerr_arr, zmin=0.0, zmax=1.5, zerr_min=0.02, zerr_max=0.5, nzbins=100, nzerrbins=49, niter=100, sigma=2):
        r"""Photo-z model obtained by Richardson-Lucy deconvolution.

        Now superseded(?) by RegulatedDeconvolver, see below.
        Source: Selim's ``pipeline_getsurrogates_selim.ipynb``, Nov 2024.

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
        """KSZ P(k) pipeline.

        Runnable either by calling the run() method, or from the command line with:
           python -m kszx <input_dir> <output_dir>

        Reminder: input directory must contain files { params.yml, galaxies.h5, randoms.h5,
        bounding_box.pkl }. The pipeline will populate the output directory with files
        { params.yml, pk_data.npy, pk_surrogates.npy }. (For details of these file formats,
        see one of the README files that Kendrick or Selim have lying around.)
        """
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.nsurr = nsurr
        
        # Create output_dir and 'tmp' subdir
        os.makedirs(f'{output_dir}/tmp', exist_ok=True)

        with open(f'{input_dir}/params.yml', 'r') as f:
            params = yaml.safe_load(f)

            self.surr_bg = params['surr_bg']
            self.surr_ic_nbins = params['surr_ic_nbins']
            self.spin0_hack = params['spin0_hack']
            assert self.spin0_hack in [ True, False ]

            self.kmax = params['kmax']
            self.nkbins = params['nkbins']
            self.kbin_edges = np.linspace(0, self.kmax, self.nkbins+1)
            self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.

        self.box = io_utils.read_pickle(f'{input_dir}/bounding_box.pkl')
        self.kernel = 'cubic'   # FIXME hardcoded for now
        self.deltac = 1.68      # FIXME hardcoded for now

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
    def rcat_xyz_obs(self):
        return self.rcat.get_xyz(self.cosmo, 'zobs')
        
    @functools.cached_property
    def surrogate_factory(self):
        # FIXME needs comment
        surr_ngal_mean = self.gcat.size
        surr_ngal_rms = 4 * np.sqrt(self.gcat.size)  # 4x Poisson
        return SurrogateFactory(self.box, self.cosmo, self.rcat, surr_ngal_mean, surr_ngal_rms, 'ztrue')

    
    @functools.cached_property
    def window_function(self):
        nrand = self.rcat.size
        rweights = getattr(self.rcat, 'weight_zerr', np.ones(nrand))
        vweights = getattr(self.rcat, 'vweight_zerr', np.ones(nrand))
        rsum = np.sum(rweights)
        vsum = np.sum(vweights)

        footprints = [
            core.grid_points(self.box, self.rcat_xyz_obs, rweights, kernel=self.kernel, fft=True, compensate=True, wscal = 1.0/rsum),
            core.grid_points(self.box, self.rcat_xyz_obs, vweights * self.rcat.bv_90, kernel=self.kernel, fft=True, compensate=True, wscal = 1.0/vsum),
            core.grid_points(self.box, self.rcat_xyz_obs, vweights * self.rcat.bv_150, kernel=self.kernel, fft=True, compensate=True, wscal = 1.0/vsum)
        ]
        
        wf = wfunc_utils.compute_wcrude(self.box, footprints)
        return wf

    
    @functools.cached_property
    def pse(self):
        """Returns a KszPSE object."""
        
        print('Initializing KszPSE: this will take a few minutes')
        
        # FIXME needs comment
        surr_ngal_mean = self.gcat.size
        surr_ngal_rms = 4 * np.sqrt(self.gcat.size)  # 4x Poisson
        
        rweights = getattr(self.rcat, 'weight_zerr', None)
        vweights = getattr(self.rcat, 'vweight_zerr', None)
        
        pse = KszPSE(
            box = self.box, 
            cosmo = self.cosmo, 
            randcat = self.rcat, 
            kbin_edges = self.kbin_edges,
            surr_ngal_mean = surr_ngal_mean,
            surr_ngal_rms = surr_ngal_rms,
            surr_bg = self.surr_bg,
            rweights = rweights,
            nksz = 2,
            ksz_rweights = vweights,
            ksz_bv = [ self.rcat.bv_90, self.rcat.bv_150 ],
            ksz_tcmb_realization = [ self.rcat.tcmb_90, self.rcat.tcmb_150 ],
            ztrue_col = 'ztrue',
            zobs_col = 'zobs',
            surr_ic_nbins = self.surr_ic_nbins,
            spin0_hack = self.spin0_hack
        )
        
        print('KszPSE initialization done')
        return pse

    
    def get_pk_data(self, run=False, force=False):
        """Returns a shape (3,3,nkbins) array.

        If run=False, then this function expects the P(k) file to be on disk from a previous pipeline run.
        If run=True, then the P(k) file will be computed if it is not on disk.
        """

        if (not force) and os.path.exists(self.pk_data_filename):
            return io_utils.read_npy(self.pk_data_filename)
        
        if not (run or force):
            raise RuntimeError(f'Kpipe.get_pk_data(): run=force=False was specified, and file {self.pk_data_filename} not found')

        print('get_pk_data(): running\n', end='')

        # FIXME mean subtraction moved into KszPSE -- need to make this less confusing
        # t90 = utils.subtract_binned_means(self.gcat.tcmb_90, self.gcat.z, nbins=25)
        # t150 = utils.subtract_binned_means(self.gcat.tcmb_150, self.gcat.z, nbins=25)

        gweights = getattr(self.gcat, 'weight_zerr', None)
        vweights = getattr(self.gcat, 'vweight_zerr', None)

        pk_data = self.pse.eval_pk(
            gcat = self.gcat,
            gweights = gweights,
            ksz_gweights = vweights,
            ksz_bv = [ self.gcat.bv_90, self.gcat.bv_150 ], 
            ksz_tcmb = [ self.gcat.tcmb_90, self.gcat.tcmb_150 ],
            zobs_col = 'z'
        )

        io_utils.write_npy(self.pk_data_filename, pk_data)
        return pk_data

    
    def get_pk_data2(self, run=False, force=False):
        if (not force) and os.path.exists(self.pk_data_filename):
            return io_utils.read_npy(self.pk_data_filename)

        if not (run or force):
            raise RuntimeError(f'Kpipe.get_pk_data2(): run=force=False was specified, and file {self.pk_data_filename} not found')
        
        print('get_pk_data2(): running\n', end='')
        
        gweights = getattr(self.gcat, 'weight_zerr', np.ones(self.gcat.size))
        rweights = getattr(self.rcat, 'weight_zerr', np.ones(self.rcat.size))
        vweights = getattr(self.gcat, 'vweight_zerr', np.ones(self.gcat.size))
        gcat_xyz = self.gcat.get_xyz(self.cosmo)

        # Mean subtraction.
        coeffs_v90 = utils.subtract_binned_means(vweights * self.gcat.tcmb_90, self.gcat.z, nbins=25)
        coeffs_v150 = utils.subtract_binned_means(vweights * self.gcat.tcmb_150, self.gcat.z, nbins=25)

        # Note spin=1 on the vr fields.
        fourier_space_maps = [
            core.grid_points(self.box, gcat_xyz, gweights, self.rcat_xyz_obs, rweights, kernel=self.kernel, fft=True, compensate=True),
            core.grid_points(self.box, gcat_xyz, coeffs_v90, kernel=self.kernel, fft=True, spin=1, compensate=True),
            core.grid_points(self.box, gcat_xyz, coeffs_v150, kernel=self.kernel, fft=True, spin=1, compensate=True)
        ]

        # Rescale window function
        w = np.array([ np.sum(gweights), np.sum(vweights) ])
        wf = self.window_function * w[[0,1,1],None] * w[None,[0,1,1]]
        
        pk = core.estimate_power_spectrum(self.box, fourier_space_maps, self.kbin_edges)
        pk /= wf[:,:,None]
        
        io_utils.write_npy(self.pk_data_filename, pk)
        return pk

    
    def get_pk_surrogate(self, isurr, run=False, force=False):
        """Returns a shape (6,6,nkbins) array.
        
        If run=False, then this function expects the P(k) file to be on disk from a previous pipeline run.
        If run=True, then the P(k) file will be computed if it is not on disk.
        """
        
        fname = self.pk_single_surr_filenames[isurr]
        
        if (not force) and os.path.exists(fname):
            return io_utils.read_npy(fname)

        if not (run or force):
            raise RuntimeError(f'Kpipe.get_pk_surrogate(): run=False was specified, and file {fname} not found')

        print(f'get_pk_surrogate({isurr}): running\n', end='')
        
        self.pse.simulate_surrogate()
        
        for sv in [ self.pse.Sv_noise, self.pse.Sv_signal ]:
            assert sv.shape ==  (2, self.rcat.size)
            for j in range(2):
                sv[j,:] = utils.subtract_binned_means(sv[j,:], self.rcat.zobs, nbins=25)
    
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

    
    def get_pk_surrogate2(self, isurr, ngal=None, delta=None, M=None, ug=None, run=False, force=False):
        fname = self.pk_single_surr_filenames[isurr]
        
        if (not force) and os.path.exists(fname):
            return io_utils.read_npy(fname)

        if not (run or force):
            raise RuntimeError(f'Kpipe.get_pk_surrogate2(): run=False was specified, and file {fname} not found')

        print(f'get_pk_surrogate({isurr}): running\n', end='')

        zobs = self.rcat.zobs
        nrand = self.rcat.size
        rweights = getattr(self.rcat, 'weight_zerr', np.ones(nrand))
        vweights = getattr(self.rcat, 'vweight_zerr', np.ones(nrand))

        if ug is None:
            ug = np.random.normal(size=nrand)
            
        self.surrogate_factory.simulate_surrogate(ngal=ngal, delta=delta, M=M)

        ngal = self.surrogate_factory.ngal
        eta_rms = np.sqrt((nrand/ngal) - (self.surr_bg**2 * self.surrogate_factory.sigma2) * self.surrogate_factory.D**2)
        eta = ug * eta_rms

        # Coefficient arrays.
        Sg = (ngal/nrand) * rweights * (self.surr_bg * self.surrogate_factory.delta + eta)
        dSg_dfnl = (ngal/nrand) * rweights * (2 * self.deltac) * (self.surr_bg-1) * self.surrogate_factory.phi
        Sv90_noise = vweights * self.surrogate_factory.M * self.rcat.tcmb_90
        Sv150_noise = vweights * self.surrogate_factory.M * self.rcat.tcmb_150
        Sv90_signal = (ngal/nrand) * vweights * self.rcat.bv_90 * self.surrogate_factory.vr
        Sv150_signal = (ngal/nrand) * vweights * self.rcat.bv_150 * self.surrogate_factory.vr

        # Mean subtraction.
        Sg = utils.subtract_binned_means(Sg, zobs, self.surr_ic_nbins)
        dSg_dfnl = utils.subtract_binned_means(dSg_dfnl, zobs, self.surr_ic_nbins)
        Sv90_noise = utils.subtract_binned_means(Sv90_noise, zobs, nbins=25)   # FIXME hardcoded
        Sv90_signal = utils.subtract_binned_means(Sv90_signal, zobs, nbins=25)
        Sv150_noise = utils.subtract_binned_means(Sv150_noise, zobs, nbins=25)
        Sv150_signal = utils.subtract_binned_means(Sv150_signal, zobs, nbins=25)

        # Note spin=1 on the vr fields.
        fourier_space_maps = [
            core.grid_points(self.box, self.rcat_xyz_obs, Sg, kernel=self.kernel, fft=True, spin=0, compensate=True),
            core.grid_points(self.box, self.rcat_xyz_obs, dSg_dfnl, kernel=self.kernel, fft=True, spin=0, compensate=True),
            core.grid_points(self.box, self.rcat_xyz_obs, Sv90_noise, kernel=self.kernel, fft=True, spin=1, compensate=True),
            core.grid_points(self.box, self.rcat_xyz_obs, Sv90_signal, kernel=self.kernel, fft=True, spin=1, compensate=True),
            core.grid_points(self.box, self.rcat_xyz_obs, Sv150_noise, kernel=self.kernel, fft=True, spin=1, compensate=True),
            core.grid_points(self.box, self.rcat_xyz_obs, Sv150_signal, kernel=self.kernel, fft=True, spin=1, compensate=True)
        ]

        # Rescale window function.
        w = np.array([ ngal * np.mean(rweights), ngal * np.mean(vweights) ])
        wf = self.window_function * w[[0,1,1],None] * w[None,[0,1,1]]
        wf = wf[[0,0,1,1,2,2],:][:,[0,0,1,1,2,2]]

        pk = core.estimate_power_spectrum(self.box, fourier_space_maps, self.kbin_edges)
        pk /= wf[:,:,None]
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


####################################################################################################


class PipelineOutdir:
    def __init__(self, dirname, nsurr=None):
        """
        PipelineOutdir: a helper class for postprocessing/plotting pipeline outputs.        
        (Note: there is a separate class 'PgvLikelihood' for MCMCs and parameter fits, see below)

        The constructor reads the following files::
        
          {dirname}/params.yml         # for nkbins, kmax
          {dirname}/pk_data.npy        # shape (3, 3, nkbins)
          {dirname}/pk_surrogates.npy  # shape (nsurr, 6, 6, nkbins)
          
        Constructor arguments:

          - dirname (string): name of pipeline output directory.
         
          - nsurr (integer or None): this is a hack for running on an incomplete
            pipeline. If specified, then {dirname}/pk_surr.npy is not read.
            Instead we read files of the form {dirname}/tmp/pk_surr_{i}.npy.
        """

        filename = f'{dirname}/params.yml'
        print(f'Reading {filename}')
        
        with open(filename, 'r') as f:
            params = yaml.safe_load(f)

        kmax = params['kmax']
        nkbins = params['nkbins']
        surr_bg = params['surr_bg']
        kbin_edges = np.linspace(0, kmax, nkbins+1)
        kbin_centers = (kbin_edges[1:] + kbin_edges[:-1]) / 2.

        pk_data = io_utils.read_npy(f'{dirname}/pk_data.npy')
        if pk_data.shape != (3,3,nkbins):
            raise RuntimeError(f'Got {pk_data.shape=}, expected (3,3,nkbins) where {nkbins=}')

        if nsurr is None:
            pk_surr = io_utils.read_npy(f'{dirname}/pk_surrogates.npy')
            if (pk_surr.ndim != 4) or (pk_surr.shape[1:] != (6,6,nkbins)):
                raise RuntimeError(f'Got {pk_surr.shape=}, expected (nsurr,6,6,nkbins) where {nkbins=}')

        else:
            assert nsurr >= 1
            pk_surr = [ ]
            
            for i in range(nsurr):
                pk_surrogate = io_utils.read_npy(f'{dirname}/tmp/pk_surr_{i}.npy')
                if pk_surrogate.shape != (6,6,nkbins):
                    raise RuntimeError(f'Got {pk_surrogate.shape=}, expected (6,6,nkbins) where {nkbins=}')
                pk_surr.append(pk_surrogate)
                
            pk_surr = np.array(pk_surr)

        self.k = kbin_centers
        self.nkbins = nkbins
        self.kmax = kmax
        self.dk = kmax / nkbins
        
        self.pk_data = pk_data
        self.pk_surr = np.array(pk_surr)
        self.nsurr = self.pk_surr.shape[0]
        self.surr_bg = surr_bg


    def pgg_data(self):
        """Returns shape (nkbins,) array containing P_{gg}^{data}."""
        return self.pk_data[0,0,:]
        
    def pgg_mean(self, fnl=0):
        """Returns shape (nkbins,) array, containing <P_{gg}^{surr}>."""
        return np.mean(self._pgg_surr(fnl), axis=0)

    def pgg_rms(self, fnl=0):
        """Returns shape (nkbins,) array, containing sqrt(Var(P_{gg}^{surr}))."""
        return np.sqrt(np.var(self._pgg_surr(fnl), axis=0))

    
    def pgv_data(self, field):
        """Returns shape (nkbins,) array containing P_{gv}^{data}.

        The 'field' argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - field=[1,0] for 90 GHz
           - field=[0,1] for 150 GHz
           - field=[1,-1] for null (90-150) GHz.
        """
        field = self._check_field(field)
        return field[0]*self.pk_data[0,1] + field[1]*self.pk_data[0,2]
        
    def pgv_mean(self, field, fnl, bv):
        """Returns shape (nkbins,) array containing <P_{gv}^{surr}>.

        The 'field' argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - field=[1,0] for 90 GHz
           - field=[0,1] for 150 GHz
           - field=[1,-1] for null (90-150) GHz.
        """
        return np.mean(self._pgv_surr(field,fnl,bv), axis=0)

    def pgv_rms(self, field, fnl, bv):
        """Returns shape (nkbins,) array containing sqrt(Var(P_{gv}^{surr})).

        The 'field' argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - field=[1,0] for 90 GHz
           - field=[0,1] for 150 GHz
           - field=[1,-1] for null (90-150) GHz.
        """
        return np.sqrt(np.var(self._pgv_surr(field,fnl,bv), axis=0))


    def pvv_data(self, field):
        """Returns shape (nkbins,) array containing P_{vv}^{data}.

        The 'field' argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - field=[1,0] for 90 GHz
           - field=[0,1] for 150 GHz
           - field=[1,-1] for null (90-150) GHz.
        """
        field = self._check_field(field)
        t = field[0]*self.pk_data[1,:] + field[1]*self.pk_data[2,:]  # shape (3,)
        return field[0]*t[1] + field[1]*t[2]
        
    def pvv_mean(self, field, bv):
        """Returns shape (nkbins,) array containing < P_{vv}^{data} >.

        The 'field' argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - field=[1,0] for 90 GHz
           - field=[0,1] for 150 GHz
           - field=[1,-1] for null (90-150) GHz.
        """
        return np.mean(self._pvv_surr(field,bv), axis=0)
        
    def pvv_rms(self, field, bv):
        """Returns shape (nkbins,) array containing sqrt(var(P_{vv}^{data})).

        The 'field' argument is a length-2 vector, selecting a linear combination
        of the 90+150 GHz velocity reconstructions. For example:
        
           - field=[1,0] for 90 GHz
           - field=[0,1] for 150 GHz
           - field=[1,-1] for null (90-150) GHz.
        """        
        return np.sqrt(np.var(self._pvv_surr(field,bv), axis=0))


    def _check_field(self, field):
        """Checks that 'field' is a 1-d array of length 2."""
        field = np.array(field, dtype=float)
        if field.shape != (2,):
            raise RuntimeError(f"Expected 'field' argument to be a 1-d array of length 2, got {field.shape=}")
        return field
    
    def _pgg_surr(self, fnl=0):
        """Returns shape (nsurr, nkbins) array, containing P_{gg} for each surrogate"""
        pgg = self.pk_surr[:,:2,:2,:]             # shape (nsurr, 2, 2, nkbins)
        pgg = pgg[:,0,:,:] + fnl * pgg[:,1,:,:]   # shape (nsurr, 2, nkbins)
        return pgg[:,0,:] + fnl * pgg[:,1,:]      # shape (nsurr, nkbins)
    
    def _pgv_surr(self, field, fnl, bv):
        """Returns shape (nsurr, nkbins) array, containing P_{gv} for each surrogate"""
        field = self._check_field(field)
        pgv = self.pk_surr[:,:2,2:,:]                                # shape (nsurr, 2, 4, nkbins)
        pgv = pgv[:,0,:,:] + fnl * pgv[:,1,:,:]                      # shape (nsurr, 4, nkbins)
        pgv = field[0] * pgv[:,0:2,:] + field[1] * pgv[:,2:4,:]      # shape (nsurr, 2, nkbins)
        return pgv[:,0,:] + bv * pgv[:,1,:]                          # shape (nsurr, nkbins)

    def _pvv_surr(self, field, bv):
        """Returns shape (nsurr, nkbins) array, containing P_{vv} for each surrogate"""
        field = self._check_field(field)
        pvv = self.pk_surr[:,2:,2:,:]                                  # shape (nsurr, 4, 4, nkbins)
        pvv = field[0] * pvv[:,0:2,:,:] + field[1] * pvv[:,2:4,:,:]    # shape (nsurr, 2, 4, nkbins)
        pvv = field[0] * pvv[:,:,0:2,:] + field[1] * pvv[:,:,2:4,:]    # shape (nsurr, 2, 2, nkbins)
        pvv = pvv[:,0,:,:] + bv * pvv[:,1,:,:]                         # shape (nsurr, 2, nkbins)
        return pvv[:,0,:] + bv * pvv[:,1,:]                            # shape (nsurr, nkbins)


####################################################################################################


class PgvLikelihood:
    def __init__(self, data, surrs, bias_matrix, jeffreys_prior=False):
        """Likelihood function for one or more power spectra of the form P_{gv}(k).

        The PgvLikelihood constructor has an abstract syntax, which may not be the
        most convenient. Instead of calling the constructor directly, you may want
        to call the from_pipeline_outdir() static method as follows::

           dirname = ...   # directory name containing pipeline outputs
           pout = PipelineOutputs(dirname)

           # The 'fields=[[1,0]]' argument selects 90 GHz.
           # See the from_pipeline_outdir() docstring for more examples.
           lik = PgvLikelihood.from_pipeline_outdir(pipeline_outputs, fields=[[1,0]], nkbins=10)

        The rest of this docstring describes the PgvLikelihood constructor syntax
        (even though, as explained above, you probably don't want to call the constructor
        directly!) First we define::

          B = number of bias parameters in the MCMC (usually 1)
          V = number of velocity reconstructions (usually 1)
          K = number of k-bins
          S = number of surrogate sims
          D = V*K = number of components in "data vector".

        Note the distinction between a 90+150 GHz "sum" fit (V=1), and a 90+150 GHz
        "joint" fit (V=2). In the first case, we add the 90+150 bandpowers and construct
        a likelihood based on their sum, and in the second case we do a joint fit to
        both sets of bandpowers (i.e. doubling the size of the data vector and
        covariance matrices).
        
        Then the constructor arguments are as follows:

          - data: P_{gv} "data" bandpowers, as array of shape (V,K).
        
          - surrs: P_{gv} surrogate bandpowers, as array of shape (S,2,2,V,K).
              where the first length-2 index is fnl exponent {0,1}
              and the second length-2 index is bias exponent {0,1}

          - bias_matrix: array of shape (B,V), where B <= V. This gives the
            correspondence between bias parameters and velocity reconstruction.
            There are basically 3 cases of interest here:

              - single field (V=1):
                  bias_matrix=[[1]]
        
              - joint analysis (V=2), bias assumed to be the same for both freqs:
                  bias_matrix = [[1,1]]

              - joint analysis (V=2) with two independent bias parameters:
                  bias_matrix = [[1,0],[0,1]]
        """

        data = np.asarray(data, dtype=float)
        surrs = np.asarray(surrs, dtype=float)
        bias_matrix = np.asarray(bias_matrix, dtype=float)
        
        if data.ndim != 2:
            raise RuntimeError(f'got {data.shape=}, expected (V,K)')
        if (surrs.ndim != 5) or (surrs.shape[1:3] != (2,2)):
            raise RuntimeError(f'got {surrs.shape=}, expected (S,2,2,V,K)')
        if bias_matrix.ndim != 2:
            raise RuntimeError(f'got {bias_matrix.shape=}, expected (B,V)')

        if data.shape != surrs.shape[3:]:
            raise RuntimeError(f'data/surrs have inconsistent shapes (expected (V,K) and (S,2,2,V,K), got {data.shape} and {surr.shape})')
        if data.shape[0] != bias_matrix.shape[1]:
            raise RuntimeError(f'data/bias_matrix have inconsistent shapes (expected (V,K) and (B,V), got {data.shape} and {bias_matrix.shape})')
        if bias_matrix.shape[0] > bias_matrix.shape[1]:
            raise RuntimeError(f'expected B <= V, got (B,V)={bias_matrix.shape}')

        self.B = bias_matrix.shape[0]
        self.V = bias_matrix.shape[1]
        self.K = data.shape[1]
        self.S = surrs.shape[0]
        self.D = self.V * self.K
        
        self.data = data                                 # shape (V,K)
        self.data_vector = np.reshape(data, (self.D,))   # shape (D,)
        self.surrs = surrs
        self.bias_matrix = bias_matrix
        self.jeffreys_prior = jeffreys_prior

        self._init_fast_likelihood()


    @staticmethod
    def from_pipeline_outdir(pout, fields, nkbins, multi_bias=None, jeffreys_prior=None):
        """
        Constructs a PgvLikelhood from a PipelineOutput object. Usually more convenient
        than calling the PgvLikeihood constructor directly.

        The 'pout' argument is a PipelineOutdir object.
        
        The 'fields' argument is a V-by-2 matrix. Each row of the matrix selects a linear
        combination of the 90+150 GHz velocity reconstructions. For example:
        
           - fields = [[1,0]] for 90 GHz analysis
           - fields = [[0,1]] for 150 GHz analysis
           - fields = [[1,1]] for (90+150) "sum map" analysis
           - fields = [[1,-1]] for null (90-150) "null map" analysis
           - fields = [[1,0],[0,1]] for joint analysis with both (90+150 GHz) sets of bandpowers

        The 'multi_bias' argument only needed for a joint analysis (i.e. V > 1) and determines
        whether each freq channel has an independent bias parameter (multi_bias=True), or whether
        all frequency channels have the same bias.
        """
        
        assert isinstance(pout, PipelineOutdir)
        assert 4 <= nkbins <= pout.nkbins
        
        fields = np.array(fields)
        if (fields.ndim != 2) or (fields.shape[0] < 1) or (fields.shape[1] != 2):
            raise RuntimeError(f"PgvLikelihood: expected 'fields' arg to have shape (V,2), got shape {fields.shape}")

        K = nkbins
        S = pout.nsurr
        V = fields.shape[0]

        if (V > 1) and (multi_bias is None):
            raise RuntimeError(f"The 'multi_bias' argument is required if nfields > 1.")

        # PgvLikelihood constructor expects data array of shape (V,K)
        pgv_data = pout.pk_data[0,1:3,:K]      # shape (2,K)
        pgv_data = [ (f[0]*pgv_data[0,:] +f[1]*pgv_data[1,:]) for f in fields ]  # shape (V,K)

        # PgvLikelihood constructor expects surrogate array of shape (S,2,2,V,K).
        pgv_surr = pout.pk_surr[:,0:2,2:6,:K]           # shape (S,2,4,K)
        pgv_surr = np.reshape(pgv_surr, (S,2,2,2,K))    # shape (S,2,2,2,K), length-2 indices are (fnl_exponent, freq, bv_exponent)
        pgv_surr = [ (f[0]*pgv_surr[:,:,0,:,:] + f[1]*pgv_surr[:,:,1,:,:]) for f in fields ]  # shape (V,S,2,2,K)
        pgv_surr = np.transpose(pgv_surr, (1,2,3,0,4))  # shape (S,2,2,V,K)

        bias_matrix = np.identity(V) if multi_bias else np.ones((1,V))
        return PgvLikelihood(pgv_data, pgv_surr, bias_matrix, jeffreys_prior)

    
    def specialize_surrogates(self, fnl, bv, flatten):
        """
        Returns a shape (S,V,K) array, by "specializing" surrogates to (fnl, bv).
        Convenient but slow! Used in many places, but not fast_likelihood().
        """

        S, K, V, D = self.S, self.K, self.V, self.D
        
        fnl = float(fnl)        
        b = self._validate_bv(bv)        # shape (B,)
        b = np.dot(b, self.bias_matrix)  # shape (V,)
        b = np.reshape(b, (1,V,1))       # shape (1,V,1)

        # Apply fnl, obtaining shape (S,2,V,K)
        s = self.surrs[:,0,:,:,:] + fnl * self.surrs[:,1,:,:,:]

        # Apply bv, obtaining shape (S,V,K)
        s = s[:,0,:,:] + b * s[:,1,:,:]

        # Return either shape (S,V,K) or shape (S,D), depending on whether flatten=True.
        return np.reshape(s,(S,D)) if flatten else s
                
        
    def slow_mean_and_cov(self, fnl, bv):
        """Returns (mean, cov) where mean.shape=(D,) and cov.shape=(D,D)."""
        
        s = self.specialize_surrogates(fnl, bv, flatten=True)  # shape (S,D)
        mean = np.mean(s, axis=0)       # shape (D,)
        cov = np.cov(s, rowvar=False)   # shape (D, D)
        return mean, cov

    
    def slow_mean_and_cov_gradients(self, fnl, bv):
        """Returns grad(mu), grad(cov). Used to compute Jeffreys prior.

        Both gradients are with respect to (fnl,bv) and are represented as arrays 
        with an extra length-(B+1) axis, i.e.

          mu.shape = (D,)    =>  grad_mu.shape = (B+1,D)
          cov.shape = (D,D)  =>  grad_cov.shape = (B+1,D,D)
          
        Uses boneheaded algorithm: since mu, C are at most quadratic in fNL (and bv),
        naive finite difference is exact (and independent of step sizes).
        """

        B, D = self.B, self.D

        fnl = float(fnl)        
        bv = self._validate_bv(bv)   # shape (B,)

        # Parameter vectors of shape (B+1,)
        x0 = np.concatenate(((fnl,), bv))
        dx = np.concatenate(((50,), np.full(B,0.1)))
        
        mu_p = np.zeros((B+1, D))
        mu_n = np.zeros((B+1, D))
        cov_p = np.zeros((B+1, D, D))
        cov_n = np.zeros((B+1, D, D))

        for i in range(B+1):
            xp = np.copy(x0)
            xn = np.copy(x0)
            xp[i] += dx[i]
            xn[i] -= dx[i]

            mu_p[i,:], cov_p[i,:,:] = self.slow_mean_and_cov(xp[0], xp[1:])
            mu_n[i,:], cov_n[i,:,:] = self.slow_mean_and_cov(xn[0], xn[1:])
   
        grad_mu = (mu_p - mu_n) / (2 * dx.reshape((B+1,1)))
        grad_cov = (cov_p - cov_n) / (2 * dx.reshape((B+1,1,1)))
 
        return grad_mu, grad_cov
        
        
    def slow_log_likelihood(self, fnl, bv):
        """Returns the scalar quantity log(L)."""
        
        mean, cov = self.slow_mean_and_cov(fnl, bv)

        x = self.data_vector - mean
        cinv = np.linalg.inv(cov)
        sign, logabsdet = np.linalg.slogdet(cov)
        assert sign == 1

        # log L = -(1/2) log(det C) - (1/2) x^T C^{-1} x
        logL = -0.5 * np.dot(x, np.dot(cinv,x))
        logL -= 0.5 * logabsdet

        if self.jeffreys_prior:
            grad_mu, grad_cov = self.slow_mean_and_cov_gradients(fnl, bv)
            cinv_dmu = [ np.dot(cinv,dmu) for dmu in grad_mu ]
            cinv_dc = [ np.dot(cinv,dc) for dc in grad_cov ]

            # (B+1)-by-(B+1) Fisher matrix            
            B = self.B
            f = np.zeros((B+1,B+1))
            for i in range(B+1):
                for j in range(B+1):
                    f[i,j] += np.dot(grad_mu[i], cinv_dmu[j])  # no factor (1/2)
                    f[i,j] += 0.5 * np.trace(np.dot(cinv_dc[i], cinv_dc[j]))

            # Jeffreys prior is equivalent to including sqrt(det(F)) in the likelihood.
            sign, logabsdet_F = np.linalg.slogdet(f)
            logL += 0.5 * logabsdet_F
            assert sign == 1
            
        return logL

    
    def run_mcmc(self, nwalkers=8, nsamples=10000, discard=1000, thin=5):
        """
        Initializes self.samples to an array of shape (N,2) where N is large,
        and the length-2 axis represents {fnl,bv}.
        """

        import emcee
        print(f'MCMC start: {nwalkers=}, {nsamples=}, {discard=}, {thin=}')

        x0 = np.zeros((nwalkers, self.B+1))
        x0[:,0] = np.random.uniform(-50, 50, size=nwalkers)  # fnl
        x0[:,1:] = np.random.uniform(0.5, 1.0, size=(nwalkers,self.B))  # bv

        logL = lambda x: self.fast_log_likelihood(x[0], x[1:])
        sampler = emcee.EnsembleSampler(nwalkers, self.B+1, logL)
        sampler.run_mcmc(x0, nsamples)
        self.samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        
        print('MCMC done. To see the results, call the show_mcmc() method.')


    def show_mcmc(self, title=None):
        """Makes a corner plot from MCMC results."""

        if not hasattr(self, 'samples'):
            raise RuntimeError('Must call ')
        
        import corner
        
        fig = corner.corner(self.samples, bins=100, range=(0.99,0.99), labels=[r'$f_{NL}$',r'$b_v$'])
        if title is not None:
            fig.suptitle(title)
        
        plt.show()

        fnl_samples = self.samples[:,0]  # shape (nsamp,)
        bv_samples = self.samples[:,1:]  # shape (nsamp,B)
        qlevels = [ 0.025, 0.16, 0.5, 0.84, 0.975 ]
        
        fnl_quantiles = np.quantile(fnl_samples, qlevels)
        for q,fnl in zip(qlevels, fnl_quantiles):
            s = f'  ({(fnl-fnl_quantiles[2]):+.03f})' if (q != 0.5) else ''
            print(f'{(100*q):.03f}% quantile: {fnl=:.03f}{s}')

        for b in range(self.B):
            bv_quantiles = np.quantile(bv_samples[:,b], qlevels)
            print(f'\nBias parameter {b}')
            for q,bv in zip(qlevels, bv_quantiles):
                s = f'  ({(bv-bv_quantiles[2]):+.03f})' if (q != 0.5) else ''
                print(f'{(100*q):.03f}% quantile: {bv=:.03f}{s}')

        print(f'\nSNR: {self.compute_snr():.03f}')


    def analyze_chi2(self, fnl, bv, ddof=None):
        """Do model parameters (fnl,bv) fit the data? Returns (chi2, ndof, p-value).

        The 'ddof' argument is used to compute the number of degrees of freedom:
            ndof = nkbins - ddof

        If ddof=None, then it will be equal to the number of nonzero (fnl, bias) params.
        """
        
        fnl = float(fnl)        
        bv = self._validate_bv(bv)   # shape (B,)
        mean, cov = self.slow_mean_and_cov(fnl, bv)

        if ddof is None:
            ddof = 1 if (fnl != 0) else 0
            ddof += np.count_nonzero(bv)
        
        x = self.data_vector - mean
        chi2 = np.dot(x, np.linalg.solve(cov,x))
        ndof = self.K - ddof
        pte = scipy.stats.chi2.sf(chi2, ndof)
        
        return chi2, ndof, pte
        
    
    def fit_fnl_and_bv(self, fnl0=0, bv0=0.3):
        """Returns (fnl, bv) obtained by maximizing joint likelihood."""

        x0 = np.zeros(self.B+1)
        x0[0] = fnl0
        x0[1:] = bv0

        f = lambda x: -self.fast_log_likelihood(x[0], x[1:])  # note minus sign
        result = scipy.optimize.minimize(f, x0, method='Nelder-Mead')
        
        fnl = result.x[0]
        bv = result.x[1:]
        return fnl, bv        

        
    def fit_bv(self, fnl=0, bv0=0.3):
        """Returns bv obtained by maximizing conditional likelihood at the given fNL."""

        x0 = np.full(self.B, bv0)
        f = lambda x: -self.fast_log_likelihood(fnl, x)  # note minus sign
        result = scipy.optimize.minimize(f, x0, method='Nelder-Mead')
        
        bv = result.x
        return bv


    def compute_snr(self):
        # This implementation works even if there are multiple bias params (B > 1).
        B, D = self.B, self.D
        
        _, cov = self.slow_mean_and_cov(0, np.zeros(self.B))                # discard mean
        grad_mu, _ = self.slow_mean_and_cov_gradients(0, np.zeros(self.B))  # discard grad_cov
        m = grad_mu[1:,:]           # shape (B,D)       
        d = self.data_vector        # shape (D,)

        cinv_m = np.linalg.solve(cov, m.T)  # shape (D,B)
        h = np.dot(m, cinv_m)               # shape (B,B)
        g = np.dot(d, cinv_m)

        dchisq = np.dot(g, np.linalg.solve(h, g))
        return np.sqrt(dchisq)
        

    def _validate_bv(self, bv):
        """Helper method: converts 'bv' argument to a 1-d array of length B, and returns it."""

        bv = np.asarray(bv, dtype=float)       
        
        if (bv.ndim == 0) and (self.B == 1):
            return np.reshape(bv, (1,))
            
        if bv.shape != (self.B,):
            raise RuntimeError(f'Got {bv.shape=}, expected 1-d array of length {B=}')

        return bv

    
    ####################################################################################################
    #
    # "Fast" likelihood starts here.
    #
    # This code is completely unreadable, but there are unit tests which verify that the fast_*
    # functions are equivalent o their slow_* equivalents.


    def _init_fast_likelihood(self):
        S, V, K, D = self.S, self.V, self.K, self.D
        
        # xmu = shape (2,2,V,K), reshaped to (2,2*D).
        # Length-2 axes are fnl exponent and bv exponent.
        
        t = np.mean(self.surrs, axis=0)      # shape (2,2,D)
        self.xmu = np.reshape(t, (2,2*D))
        
        # xcov = shape (3,2,V,K,2,V,K), reshaped to (3,4*D*D)
        # Length-3 axis is fnl exponent {0,1,2}, and length-2 axes are bv exponents {0,1}.

        t = np.reshape(self.surrs, (S,4*D))      # shape (S, 4D)
        t = np.cov(t, rowvar=False)              # shape (4D,4D)
        t = np.reshape(t, (2,2*D,2,2*D))         # shape (2,2D,2,2D) where length-2 axes are fnl exponents
        t = np.array([ t[0,:,0,:], t[0,:,1,:]+t[1,:,0,:], t[1,:,1,:] ])   # shape (3,2D,2D)
        t = np.reshape(t, (3,4*D*D))         # shape (3,4D^2)
        self.xcov = np.copy(t)               # make contiguous


    def fast_mean_and_cov(self, fnl, bv, grad=False):
        """bv must be a 1-d array of length B, i.e. scalar is not allowed."""
        
        B, D, V, K = self.B, self.D, self.V, self.K
        f3 = np.array((1.0, fnl, fnl**2))

        bv = np.dot(bv, self.bias_matrix)  # shape (V,)
        bv20 = np.reshape(bv, (V,1))
        bv42 = np.reshape(bv, (1,1,V,1))
        bv50 = np.reshape(bv, (V,1,1,1,1))

        # Reminder: xmu = shape (2,2,V,K), reshaped to (2,2*D).
        mu0 = np.dot(f3[:2], self.xmu)       # shape (2*D)
        mu0 = np.reshape(mu0, (2,V,K))       # shape (2,V,K)
        mu = mu0[0,:,:] + bv20 * mu0[1,:,:]  # shape (V,K)
        mu = np.reshape(mu, (D,))

        # Reminder: xcov = shape (3,2,V,K,2,V,K), reshaped to (3,4*D*D)
        cov0 = np.dot(f3, self.xcov)                         # shape (4*D*D)
        cov0 = np.reshape(cov0, (2,V,K,2,V,K))               # shape (2,V,K,2,V,K)
        cov1 = cov0[0,:,:,:,:,:] + bv50 * cov0[1,:,:,:,:,:]  # shape (V,K,2,V,K)
        cov = cov1[:,:,0,:,:] + bv42 * cov1[:,:,1,:,:]       # shape (V,K,V,K)
        cov = np.reshape(cov, (D,D))

        if not grad:
            return mu, cov

        dmu_dfnl = np.reshape(self.xmu[1,:], (2,V,K))        # shape (2,V,K)
        dmu_dfnl = dmu_dfnl[0,:,:] + bv20 * dmu_dfnl[1,:,:]  # shape (V,K)
        dmu_dbv = mu0[1,:,:]                                 # shape (V,K)

        f2 = np.array((1.0, 2*fnl))
        dcov_dfnl = np.dot(f2, self.xcov[1:])                               # shape (4*D*D)
        dcov_dfnl = np.reshape(dcov_dfnl, (2,V,K,2,V,K))                    # shape (2,V,K,2,V,K)
        dcov_dfnl = dcov_dfnl[0,:,:,:,:,:] + bv50 * dcov_dfnl[1,:,:,:,:,:]  # shape (V,K,2,V,K)
        dcov_dfnl = dcov_dfnl[:,:,0,:,:] + bv42 * dcov_dfnl[:,:,1,:,:]      # shape (V,K,V,K)
        dcov_dbv0 = cov0[1,:,:,:,:,:]                                       # shape (V,K,2,V,K)
        dcov_dbv0 = dcov_dbv0[:,:,0,:,:] + bv42 * dcov_dbv0[:,:,1,:,:]      # shape (V,K,V,K)
        dcov_dbv1 = cov1[:,:,1,:,:]                                         # shape (V,K,V,K)
        
        mu_grad = np.zeros((B+1,V,K))     
        mu_grad[0,:,:] = dmu_dfnl
        mu_grad[1:,:,:] = self.bias_matrix.reshape((B,V,1)) * dmu_dbv.reshape((1,V,K))
        mu_grad = np.reshape(mu_grad, (B+1,D))
        
        cov_grad = np.zeros((B+1,V,K,V,K))
        cov_grad[0,:,:,:,:] = dcov_dfnl
        cov_grad[1:,:,:,:,:] = self.bias_matrix.reshape((B,V,1,1,1)) * dcov_dbv0.reshape((1,V,K,V,K))
        cov_grad[1:,:,:,:,:] += self.bias_matrix.reshape((B,1,1,V,1)) * dcov_dbv1.reshape((1,V,K,V,K))
        cov_grad = np.reshape(cov_grad, (B+1,D,D))
        
        return mu, cov, mu_grad, cov_grad


    def fast_log_likelihood(self, fnl, bv):
        if self.jeffreys_prior:
            # Need gradients
            mean, cov, grad_mean, grad_cov = self.fast_mean_and_cov(fnl, bv, grad=True)
        else:
            # No gradients needed
            mean, cov = self.fast_mean_and_cov(fnl, bv, grad=False)
        
        x = self.data_vector - mean
        l = np.linalg.cholesky(cov)
        linv_x = scipy.linalg.solve_triangular(l, x, lower=True)

        # log L = -(1/2) log(det C) - (1/2) x^T C^{-1} x
        logL = -0.5 * np.dot(linv_x, linv_x)
        logL -= np.sum(np.log(l.diagonal()))

        if self.jeffreys_prior:
            B, D = (self.B, self.D)
            linv_dmu = scipy.linalg.solve_triangular(l, grad_mean.T, lower=True)  # shape (D,2)
            f = np.dot(linv_dmu.T, linv_dmu)   # first term in 2-by-2 Fisher matrix

            # Second term in 2-by-2 Fisher matrix
            # F_{ij} = (1/2) Tr(C^{-1} dC_i C^{-1} dC_j)
            #        = (1/2) Tr(S_i S_j)  where S_i = L^{-1} dC_i L^{-T}
            
            t = grad_cov.reshape(((B+1)*D, D))
            u = scipy.linalg.solve_triangular(l, t.T, lower=True)  # shape (D, (B+1)*D)
            u = u.reshape((D*(B+1), D))
            v = scipy.linalg.solve_triangular(l, u.T, lower=True)  # shape (D,D*(B+1))
            v = v.reshape((D*D, B+1))
            f += 0.5 * np.dot(v.T, v)

            # Jeffreys prior is equivalent to including sqrt(det(F)) in the likelihood.
            sign, logabsdet_F = np.linalg.slogdet(f)
            logL += 0.5 * logabsdet_F
            assert sign == 1

        return logL

    
    ############################################  Testing  #############################################
    

    def test_fast_mean_and_cov(self):
        """Test fast_mean_and_cov(), by checking that it agrees with slow_mean_and_cov() at 10 random points."""

        for _ in range(10):
            fnl = np.random.uniform(-50, 50)
            bv = np.random.uniform(0, 1, size=(self.B,))
            
            slow_mean, slow_cov = self.slow_mean_and_cov(fnl, bv)
            slow_mean_grad, slow_cov_grad = self.slow_mean_and_cov_gradients(fnl, bv)
            fast_mean, fast_cov, fast_mean_grad, fast_cov_grad = self.fast_mean_and_cov(fnl, bv, grad=True)

            assert np.all(np.abs(slow_mean - fast_mean) < 1.0e-10)
            assert np.all(np.abs(slow_cov - fast_cov) < 1.0e-10)
            assert np.all(np.abs(slow_mean_grad - fast_mean_grad) < 1.0e-10)
            assert np.all(np.abs(slow_cov_grad - fast_cov_grad) < 1.0e-10)
    
    
    def test_fast_likelihood(self):
        """Test fast_log_likelihood(), by checking that it agrees with slow_log_likelihood() at 10 random points."""
        
        for _ in range(10):
            fnl = np.random.uniform(-50, 50)
            bv = np.random.uniform(0, 1, size=(self.B,))
            logL_slow = self.slow_log_likelihood(fnl, bv)
            logL_fast = self.fast_log_likelihood(fnl, bv)
            assert np.abs(logL_slow - logL_fast) < 1.0e-10
        
    
    @staticmethod
    def make_random():
        """Construct and return a PgvLikelihood with random (data, surrs, bias_matrix).
        
        Useful for standalone testing of 'class PgvLikelihood', in order to construct
        an "interesting" PgvLikelihood, in a situation where KSZ pipeline outputs are
        not available."""

        B = np.random.randint(1, 4)       # number of bias parameters in the MCMC
        V = np.random.randint(B, B+3)     # number of velocity reconstructions
        K = np.random.randint(5, 15)      # number of k-bins
        S = np.random.randint(100, 200)   # number of surrogate sims
        jeffreys_prior = (np.random.uniform() < 0.5)   # boolean

        data = np.random.normal(size=(V,K))
        surrs = np.random.normal(size=(S,2,2,V,K))

        # Randomly generate the bias_matrix (shape (B,V), where B <= V)
        # This is not so straightforward, since we want to avoid small SVD eigenvalues
        # for numerical stability.

        rot1 = utils.random_rotation_matrix(B)
        rot2 = utils.random_rotation_matrix(V)
        svds = np.random.uniform(1.0, 2.0, size=B)
        bias_matrix = np.dot(rot1, svds.reshape((B,1)) * rot2[:B,:])

        return PgvLikelihood(data, surrs, bias_matrix, jeffreys_prior)

        
    @staticmethod
    def run_tests():
        """Runs standalone tests of 'class PgvLikelihood'.
        (Where "standalone" means that no KSZ pipeline outputs are needed.)"""
        
        for _ in range(20):
            lik = PgvLikelihood.make_random()
            lik.test_fast_mean_and_cov()
            lik.test_fast_likelihood()

        print('PgvLikelihod.run_tests(): pass')
