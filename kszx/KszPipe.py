import os
import yaml
import shutil
import functools
import numpy as np

from . import core
from . import utils
from . import io_utils
from . import wfunc_utils

from .Catalog import Catalog
from .Cosmology import Cosmology
from .SurrogateFactory import SurrogateFactory


class KszPipe:
    def __init__(self, input_dir, output_dir):
        r"""This is the main kSZ analysis pipeline, which computes data/surrogate power spectra from catalogs.

        To run the pipeline, either create a KszPipe instance and call the :meth:`~KszPipe.run()` method,
        or run from the command line with::
        
           python -m kszx kszpipe_run [-p NUM_PROCESSES] <input_dir> <output_dir>
        
        The ``input_dir`` contains a parameter file ``params.yml`` and galaxy/random catalogs.
        The ``output_dir`` will be populated with power spectra.
        For details of what KszPipe computes, and documentation of file formats, see the sphinx docs:

          https://kszx.readthedocs.io/en/latest/kszpipe.html#kszpipe-details

        After running the pipeline, you may want to load pipeline outputs using the helper class
        :class:`~kszx.KszPipeOutdir`, or do parameter estimation using :class:`~kszx.PgvLikelihood`.
        
        High-level features:
        
          - Runs "surrogate" sims (see overleaf) to characterize the survey window function,
            determine dependence of power spectra on $(f_{NL}, b_v)$, and assign error bars
            to power spectra.
 
          - Velocity reconstruction noise is included in surrogate sims via a bootstrap procedure,
            using the observed CMB realization. This automatically incorporates noise inhomogeneity
            and "striping", and captures correlations e.g. between 90 and 150 GHz.

          - The galaxy catalog can be spectroscopic or photometric (via the ``ztrue_col`` and 
            ``zobs_col`` constructor args). Surrogate sims will capture the effect of photo-z errors.

          - The windowed power spectra $P_{gg}$, $P_{gv}$, $P_{vv}$ use a normalization which
            should be approximately correct. The normalization is an ansatz which is imperfect,
            especially on large scales, so surrogate sims should still be used to compare power
            spetra to models. Eventually, we'll implement a precise calculation of the window
            function.

          - Currently assumes one galaxy field, and two velocity reconstructions labelled
            "90" and "150" (with ACT in mind).

          - Currently, there is not much implemented for CMB foregrounds. Later, I'd like
            to include foreground clustering terms in the surrogate model (i.e. terms of the
            form $b_\delta * \delta(x)$, in addition to the kSZ term $b_v v_r(x)$), and estimate
            the $b_\delta$ biases by estimating the spin-zero $P_{gv}$ power spectrum.
        """
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output_dir and 'tmp' subdir
        os.makedirs(f'{output_dir}/tmp', exist_ok=True)

        with open(f'{input_dir}/params.yml', 'r') as f:
            params = yaml.safe_load(f)

            self.version = params['version']
            assert self.version == 1   # for now

            self.nsurr = params['nsurr']
            self.surr_bg = params['surr_bg']
            self.nzbins_gal = params['nzbins_gal']
            self.nzbins_vr = params['nzbins_vr']

            self.kmax = params['kmax']
            self.nkbins = params['nkbins']            
            self.kbin_edges = np.linspace(0, self.kmax, self.nkbins+1)
            self.kbin_centers = (self.kbin_edges[1:] + self.kbin_edges[:-1]) / 2.

        self.box = io_utils.read_pickle(f'{input_dir}/bounding_box.pkl')
        self.kernel = 'cubic'   # hardcoded for now
        self.deltac = 1.68      # hardcoded for now

        self.pk_data_filename = f'{output_dir}/pk_data.npy'
        self.pk_surr_filename = f'{output_dir}/pk_surrogates.npy'
        self.pk_single_surr_filenames = [ f'{output_dir}/tmp/pk_surr_{i}.npy' for i in range(self.nsurr) ]


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
    def window_function(self):
        print('Initializing KszPipe.window_function')
        
        nrand = self.rcat.size
        rweights = getattr(self.rcat, 'weight_gal', np.ones(nrand))
        vweights = getattr(self.rcat, 'weight_vr', np.ones(nrand))
        rsum = np.sum(rweights)
        vsum = np.sum(vweights)

        footprints = [
            core.grid_points(self.box, self.rcat_xyz_obs, rweights, kernel=self.kernel, fft=True, compensate=True, wscal = 1.0/rsum),
            core.grid_points(self.box, self.rcat_xyz_obs, vweights * self.rcat.bv_90, kernel=self.kernel, fft=True, compensate=True, wscal = 1.0/vsum),
            core.grid_points(self.box, self.rcat_xyz_obs, vweights * self.rcat.bv_150, kernel=self.kernel, fft=True, compensate=True, wscal = 1.0/vsum)
        ]
        
        wf = wfunc_utils.compute_wcrude(self.box, footprints)
        
        print('KszPipe.window_function initialized')
        return wf

    
    @functools.cached_property
    def surrogate_factory(self):
        print('Initializing KszPipe.surrogate_factory')
        
        # FIXME needs comment
        surr_ngal_mean = self.gcat.size
        surr_ngal_rms = 4 * np.sqrt(self.gcat.size)  # 4x Poisson
        sf = SurrogateFactory(self.box, self.cosmo, self.rcat, surr_ngal_mean, surr_ngal_rms, 'ztrue')

        print('KszPipe.surrogate_factory initialized')
        return sf

    
    def get_pk_data(self, run=False, force=False):
        r"""Returns a shape (3,3,nkbins) array, and saves it in ``pipeline_outdir/pk_data.npy``.

        The returned array contains auto and cross power spectra of the following fields:

          - 0: galaxy overdensity 
          - 1: kSZ velocity reconstruction $v_r^{90}$
          - 2: kSZ velocity reconstruction $v_r^{150}$
        
        If run=False, then this function expects the $P(k)$ file to be on disk from a previous pipeline run.
        If run=True, then the $P(k)$ file will be computed if it is not on disk.

        If force=True, then this function recomputes $P(k)$, even if it is on disk from a previous pipeline run.
        """
        
        if (not force) and os.path.exists(self.pk_data_filename):
            return io_utils.read_npy(self.pk_data_filename)

        if not (run or force):
            raise RuntimeError(f'KszPipe.get_pk_data2(): run=force=False was specified, and file {self.pk_data_filename} not found')
        
        print('get_pk_data2(): running\n', end='')
        
        gweights = getattr(self.gcat, 'weight_gal', np.ones(self.gcat.size))
        rweights = getattr(self.rcat, 'weight_gal', np.ones(self.rcat.size))
        vweights = getattr(self.gcat, 'weight_vr', np.ones(self.gcat.size))
        gcat_xyz = self.gcat.get_xyz(self.cosmo)

        # Mean subtraction.
        coeffs_v90 = utils.subtract_binned_means(vweights * self.gcat.tcmb_90, self.gcat.z, self.nzbins_vr)
        coeffs_v150 = utils.subtract_binned_means(vweights * self.gcat.tcmb_150, self.gcat.z, self.nzbins_vr)

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
        r"""Returns a shape (6,6,nkbins) array, and saves it in ``pipeline_outdir/tmp/pk_surr_{isurr}.npy``.
        
        The returned array contains auto and cross power spectra of the following fields, for a single surrogate:
        
          - 0: surrogate galaxy field $S_g$ with $f_{NL}=0$.
          - 1: derivative $dS_g/df_{NL}$.
          - 2: surrogate kSZ velocity reconstruction $S_v^{90}$, with $b_v=0$ (i.e. noise only).
          - 3: derivative $dS_v^{90}/db_v$.
          - 4: surrogate kSZ velocity reconstruction $S_v^{150}$, with $b_v=0$ (i.e. noise only).
          - 5: derivative $dS_v^{150}/db_v$.
        
        If run=False, then this function expects the $P(k)$ file to be on disk from a previous pipeline run.
        If run=True, then the $P(k)$ file will be computed if it is not on disk.

        If force=True, then this function recomputes $P(k)$, even if it is on disk from a previous pipeline run.
        """

        fname = self.pk_single_surr_filenames[isurr]
        
        if (not force) and os.path.exists(fname):
            return io_utils.read_npy(fname)

        if not (run or force):
            raise RuntimeError(f'KszPipe.get_pk_surrogate2(): run=False was specified, and file {fname} not found')

        print(f'get_pk_surrogate({isurr}): running\n', end='')

        zobs = self.rcat.zobs
        nrand = self.rcat.size
        rweights = getattr(self.rcat, 'weight_gal', np.ones(nrand))
        vweights = getattr(self.rcat, 'weight_vr', np.ones(nrand))
            
        self.surrogate_factory.simulate_surrogate()

        ngal = self.surrogate_factory.ngal
        eta_rms = np.sqrt((nrand/ngal) - (self.surr_bg**2 * self.surrogate_factory.sigma2) * self.surrogate_factory.D**2)
        eta = np.random.normal(scale = eta_rms)

        # Coefficient arrays.
        Sg = (ngal/nrand) * rweights * (self.surr_bg * self.surrogate_factory.delta + eta)
        dSg_dfnl = (ngal/nrand) * rweights * (2 * self.deltac) * (self.surr_bg-1) * self.surrogate_factory.phi
        Sv90_noise = vweights * self.surrogate_factory.M * self.rcat.tcmb_90
        Sv150_noise = vweights * self.surrogate_factory.M * self.rcat.tcmb_150
        Sv90_signal = (ngal/nrand) * vweights * self.rcat.bv_90 * self.surrogate_factory.vr
        Sv150_signal = (ngal/nrand) * vweights * self.rcat.bv_150 * self.surrogate_factory.vr

        # Mean subtraction.
        if self.nzbins_gal > 0:
            Sg = utils.subtract_binned_means(Sg, zobs, self.nzbins_gal)
            dSg_dfnl = utils.subtract_binned_means(dSg_dfnl, zobs, self.nzbins_gal)

        if self.nzbins_vr > 0:
            Sv90_noise = utils.subtract_binned_means(Sv90_noise, zobs, self.nzbins_vr)
            Sv90_signal = utils.subtract_binned_means(Sv90_signal, zobs, self.nzbins_vr)
            Sv150_noise = utils.subtract_binned_means(Sv150_noise, zobs, self.nzbins_vr)
            Sv150_signal = utils.subtract_binned_means(Sv150_signal, zobs, self.nzbins_vr)

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

        io_utils.write_npy(fname, pk)
        return pk


    def get_pk_surrogates(self):
        """Returns a shape (nsurr,6,6,nkins) array, and saves it in ``pipeline_outdir/pk_surrogates.npy``.
        
        The returned array contains auto and cross power spectra of the following fields, for all surrogates:

          - 0: surrogate galaxy field $S_g$ with $f_{NL}=0$.
          - 1: derivative $dS_g/df_{NL}$.
          - 2: surrogate kSZ velocity reconstruction $S_v^{90}$, with $b_v=0$ (i.e. noise only).
          - 3: derivative $dS_v^{90}/db_v$.
          - 4: surrogate kSZ velocity reconstruction $S_v^{150}$, with $b_v=0$ (i.e. noise only).
          - 5: derivative $dS_v^{150}/db_v$.

        This function only reads files from disk -- it does not run the pipeline.
        To run the pipeline, use :meth:`~kszx.KszPipe.run()`.
        """
        
        if os.path.exists(self.pk_surr_filename):
            return io_utils.read_npy(self.pk_surr_filename)

        if not all(os.path.exists(f) for f in self.pk_single_surr_filenames):
            raise RuntimeError(f'KszPipe.read_pk_surrogates(): necessary files do not exist; you need to call KszPipe.run()')

        pk = np.array([ io_utils.read_npy(f) for f in self.pk_single_surr_filenames ])
        
        io_utils.write_npy(self.pk_surr_filename, pk)
        return pk

        
    def run(self, processes=4):
        """Runs pipeline and saves results to disk, skipping results already on disk from previous runs.

        Implementation: creates a multiprocessing Pool, and calls :meth:`~kszx.KszPipe.get_pk_data()`
        and :meth:`~kszx.get_pk_surrogates()` in worker processes. (FIXME use MPI instead?)

        Can be run from the command line with::
        
           python -m kszx kszpipe_run [-p NUM_PROCESSES] <input_dir> <output_dir>
        """
        
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
            print(f'KszPipe.run(): pipeline has already been run, exiting early')
            return
        
        # Initialize window function and SurrogateFactory before creating multiprocessing Pool.
        self.window_function
        self.surrogate_factory

        # FIXME currently I don't have a good way of setting the number of processes automatically --
        # caller must adjust the number of processes to the amount of memory in the node.
        
        with utils.Pool(processes) as pool:
            l = [ ]
                
            if not have_data:
                l += [ pool.apply_async(self.get_pk_data, (True,False)) ]   # (run,force)=(True,False)
            for i in missing_surrs:
                l += [ pool.apply_async(self.get_pk_surrogate, (i,True)) ]  # (run,force)=(True,False)

            for x in l:
                x.get()

        if not have_surr:
            # Consolidates all surrogates into one file
            self.get_pk_surrogates()


####################################################################################################


class KszPipeOutdir:
    def __init__(self, dirname, nsurr=None):
        """
        KszPipeOutdir: a helper class for postprocessing/plotting pipeline outputs.        
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
