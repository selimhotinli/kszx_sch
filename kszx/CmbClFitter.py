from . import utils
from . import pixell_utils
from . import plot

from .Cosmology import Cosmology

import pixell
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


class CmbClFitter:
    def __init__(self, cosmo, cmb_map, weight_map, bl, lmin, lmax, ivar=None, iterative=False, fit_cmb_amplitude=False, fit_alpha=True, fit_lred=True, beamed_red_noise=True, alpha_fid=-3, lred_fid=1500, lpad=1000):
        """Given a CMB map, fit a power spectrum of the form C_l^{CAMB} + (power law) + (white noise).
        
        Based on utils.get_single_frequency_alms() in Alex + Mat's kSZquest software:
            https://github.com/alexlague/kSZquest/blob/main/utils.py

        **Placeholder docstring -- proper docstring coming soon.**

        Intended for fitting high-l tail of the CMB, so choose lmin ~ 10^3.

        NOTE: currently assumes that 'cmb_map' and 'weight_map' pixell maps, but could
        easily be modified to allow healpix maps -- let me know if this would be useful.
        
          - ``weight_map`` (pixell.enmap.ndmap): applied to all maps before computing
            alms or cls. Recommend including a foreground mask, multiplied by some 
            downweighting of noisy regions (e.g. hard cutoff, ivar weighting, or FKP
            weighting).

          - ``lpad``: only used if ``use_sims=True``.
            
        The constructor initializes the following members:: 

            self.lmin
            self.lmax
            self.alpha            # spectral index of red noise (currently -3)
            self.bl

            # "Theory" power spectra (weight_map not applied).
            
            self.cl_beamed_cmb    # beam-convolved, noise-free
            self.cl_red_noise     # power-law in l
            self.cl_white_noise   # constant in l
            self.cl_tot           # sum of prev 3 contributions

            # Weighted or "pseudo" power spectra, intended for plot().
            # Note: just using one sim for now. Some day I'll improve this, by
            # implementing the pseudo-Cl transfer matrix.
            
            self.pcl_beamed_cmb   # beam-convolved, noise-free, from one sim
            self.pcl_red_noise    # power spectrum of red noise, from one sim
            self.pcl_white_noise  # constant in l
            self.pcl_tot          # sum of prev 3 contributions
            self.pcl_data         # from 'cmb_map' constructor arg
            
            # Scalars
        
            self.l_knee           # value of l where white/red noise are equal
            self.uK_arcmin        # reparameterization of self.cl_white_noise
            self.ivar_uK_arcmin   # equivalent white noise level of ivar map
        """

        # Argument checking starts here.
        bl = utils.asarray(bl, 'CmbClFitter', 'bl', dtype=float)
        
        assert isinstance(cosmo, Cosmology)
        assert isinstance(cmb_map, pixell.enmap.ndmap)
        assert isinstance(weight_map, pixell.enmap.ndmap)
        assert 2 <= lmin < lmax
        assert lmin <= lred_fid < lmax
        assert alpha_fid < 0
        assert bl.ndim == 1
        assert lpad > 0
        
        if ivar is not None:
            assert isinstance(ivar, pixell.enmap.ndmap)

        beam_lmax = len(bl)-1
        assert cosmo.lmax >= (lmax + lpad)
        assert beam_lmax >= (lmax + lpad)
        
        # Argument checking ends here.
        
        self.lmin = lmin
        self.lmax = lmax
        self.lpad = lpad
        self.lred = lred_fid
        self.alpha = alpha_fid
        self.fit_lred = fit_lred
        self.fit_alpha = fit_alpha
        self.fit_cmb_amplitude = fit_cmb_amplitude
        self.beamed_red_noise = beamed_red_noise
        self.iterative = iterative
        self.bl = bl[:(lmax+lpad+1)]
        
        # self._init_white_noise() initializes the following members:
        #
        #  self.w2               # approximate normalization used to compute pseudo-Cls
        #  self.ivar_uK_arcmin   # equivalent noise level of ivar map
        
        self._init_white_noise(weight_map, ivar)

        # Must come after _init_white_noise()
        self.pcl_data = self._pcl_from_weighted_map(cmb_map * weight_map)
        
        # self._do_fit() initializes or updates the following members:
        #
        #  self.alpha           # current value of alpha
        #  self.red_ampl        # current value of A in C_l = A (l/lmin)^alpha
        #  self.white_ampl      # current value of B in C_l = B
        #  self.cmb_ampl        # multiplier of C_l^{lensed}
        #  self.red_template    # pseudo-cls corresponding to (l/lmin)^alpha, from one sim
        #  self.cmb_template    # pseudo-cls corresponding to C_l^{lensed}, from one sim
        #  self.l_knee          # value of l where white/red noise are equal
        #  self.uK_arcmin       # reparameterization of self.white_ampl

        self._do_fit(cosmo, weight_map)
        
        # Done -- just need to initialize remaining members.
        
        self.cl_beamed_cmb = self.cmb_ampl * cosmo.cltt_len[:(lmax+1)] * bl[:(lmax+1)]**2
        self.cl_red_noise = self.red_ampl * self._cl_red_noise()
        self.cl_white_noise = np.full(lmax+1, self.white_ampl)
        self.cl_tot = self.cl_beamed_cmb + self.cl_red_noise + self.cl_white_noise

        self.pcl_beamed_cmb = self.cmb_ampl * self.cmb_template
        self.pcl_red_noise = self.red_ampl * self.red_template
        self.pcl_white_noise = np.full(lmax+1, self.white_ampl)
        self.pcl_tot = self.pcl_beamed_cmb + self.pcl_red_noise + self.pcl_white_noise
        # Reminder: self.pcl_data was initialized above.
        
         
    def make_plots(self, filename_prefix=None, suffix='pdf', s=30):
        """Make three diagnostic plots, either on-screen (filename_prefix=None) or saved to disk.
        
          - Plot 1: pseudo-Cls, data vs model ``{filename_prefix}_pcl.{suffix}``
          - Plot 2: pseudo-Cls, data/model ``{filename_prefix}_pcl_ratio.{suffix}``
          - Plot 3: Model Cls, normalized to uK^2 ``{filename_prefix}_cl.{suffix}``
        """

        filename = (filename_prefix + '_pcl' + suffix) if (filename_prefix is not None) else None
        self._plot_smoothed(self.pcl_data, s, label='data', l2=True)
        self._plot_smoothed(self.pcl_tot, s, label='model (total)', l2=True)
        self._plot_smoothed(self.pcl_beamed_cmb, s, label='model (cmb)', l2=True)
        self._plot_smoothed(self.pcl_red_noise, s, label='model (red)', l2=True)
        self._plot_smoothed(self.pcl_white_noise, s, label='model (white)', l2=True)
        plt.title(r'Weighted (pseudo) $C_l$s: data vs model')
        plt.legend(loc = 'lower right')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l/(2\pi)$, arbitrary normalization')
        plt.yscale('log')
        plot.savefig(filename)

        filename = (filename_prefix + '_pcl_ratio' + suffix) if (filename_prefix is not None) else None
        self._plot_smoothed(self.pcl_data/self.pcl_tot, s, label='data/model')
        plt.title(r'(Data $C_l$) / (Model $C_l$)')
        plt.legend(loc='lower right')
        plt.xlabel(r'$l$')
        plot.savefig(filename)

        filename = (filename_prefix + '_cl' + suffix) if (filename_prefix is not None) else None
        self._plot_smoothed(self.cl_tot, s, label='model (total)', l2=True)
        self._plot_smoothed(self.cl_beamed_cmb, s, label='model (cmb)', l2=True)
        self._plot_smoothed(self.cl_red_noise, s, label='model (red)', l2=True)
        self._plot_smoothed(self.cl_white_noise, s, label='model (white)', l2=True)
        plt.title(r'Contributions to model $C_l$s (normalized to $\mu$K$^2$)')
        plt.legend(loc = 'lower right')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l/(2\pi)$ ($\mu$K$^2$)')
        plt.yscale('log')
        plot.savefig(filename)      


    def _do_fit(self, cosmo, weight_map):
        """Fits model parameters to self.pcl_data.

        Initializes or updates:

           self.alpha           # current value of alpha
           self.red_ampl        # current value of A in C_l = A (l/lmin)^alpha
           self.white_ampl      # current value of B in C_l = B
           self.cmb_ampl        # multiplier of C_l^{lensed}
           self.red_template    # pseudo-cls corresponding to (l/lmin)^alpha, from one sim
           self.cmb_template    # pseudo-cls corresponding to C_l^{lensed}, from one sim
           self.l_knee          # value of l where white/red noise are equal
           self.uK_arcmin       # reparameterization of self.white_ampl

        Assumes caller has initialized self.alpha. If we're doing iterative fitting, and this
        is not the first call to _do_fit(), then (self.red_template, self.cmb_template) will
        also be used.
        """

        lmin, lmax, lpad = self.lmin, self.lmax, self.lpad

        cl_red = self._cl_red_noise()
        cl_cmb = cosmo.cltt_len[:(lmax+lpad+1)] * self.bl**2   # beamed
        red_template = getattr(self, 'red_template', cl_red)
        cmb_template = getattr(self, 'cmb_template', cl_cmb[:(lmax+1)])
        
        # ninv = inverse variance of C_l, used to determine weighting in fit.
        # I thought it made more conceptual sense to take ninv= (2*l+1)/C_l^2, but
        # I got better results using ierr = 1. (Overall normalization is not important.)
        ninv = np.zeros(lmax+1)
        # ninv[lmin:] = np.sqrt(2*np.arange(lmax+1)+1) / self.pcl_data[lmin:]
        ninv[lmin:] = 1.0

        if False:
            # d = data vector (pcl_data - cmb_template)
            d = self.pcl_data - cmb_template
        
            # M = template matrix (ntemplates by (lmax+1))
            m = [ red_template, white_template ]
            m += [ cmb_template ] if self.fit_cmb_amplitude else []
            m = np.array(m)
            
            a = np.dot(m*ninv, m.T)
            v = np.dot(m*ninv, d)
            coeffs = np.dot(np.linalg.inv(a), v)

            self.red_ampl = coeffs[0]
            self.white_ampl = coeffs[1]
            self.cmb_ampl = (coeffs[2] + 1.0) if self.fit_cmb_amplitude else 1.0

        # Parameter ordering is (red, white, cmb-1, alpha, lred)
        x0 = [0,0]
        x0 += ([0] if self.fit_cmb_amplitude else [])
        x0 += ([self.alpha] if self.fit_alpha else [])
        x0 += ([self.lred] if self.fit_lred else [])
        
        def residual_cl(x):
            alpha = self.alpha
            lred = self.lred
            
            if self.fit_lred:
                lred = x[-1]
                x = x[:-1]

            if self.fit_alpha:
                alpha = x[-1]
                x = x[:-1]

            rcl = self.pcl_data - cmb_template

            # Treatment of red noise is a little awkward
            red_cl_ratio = self._cl_red_noise(alpha=alpha,lred=lred) / cl_red
            rcl -= x[0] * red_cl_ratio * red_template
            rcl -= x[1]   # white noise template is C_l = 1

            if self.fit_cmb_amplitude:
                rcl -= x[2] * cmb_template

            return rcl * np.sqrt(ninv)

        lsq = scipy.optimize.least_squares(residual_cl, x0)
        x1 = list(np.copy(lsq['x']))

        self.red_ampl = x1.pop(0)
        self.white_ampl = x1.pop(0)
        self.cmb_ampl = (x1.pop(0)+1) if self.fit_cmb_amplitude else 1.0
        self.alpha = x1.pop(0) if self.fit_alpha else self.alpha
        self.lred = x1.pop(0) if self.fit_lred else self.lred
        self.l_knee = lmin * (self.white_ampl / self.red_ampl)**(1/self.alpha)
        self.uK_arcmin = np.sqrt(self.white_ampl) * (60*180/np.pi)

        # Recompute 'cl_red', in case (alpha, lred) have changed.
        cl_red = self._cl_red_noise(lmax = lmax+lpad)
            
        if not hasattr(self, 'cmb_template'):
            self.cmb_template = self._pcl_from_cl(cl_cmb, weight_map)
            
        if not hasattr(self, 'red_template') or self.fit_alpha:
            self.red_template = self._pcl_from_cl(cl_red, weight_map)

        print('CmbClFitter._do_fit()')
        print(f'    alpha = {self.alpha}')
        print(f'    lred = {self.lred}')
        print(f'    cmb_ampl = {self.cmb_ampl}')
        print(f'    white_ampl = {self.white_ampl}  (uK_arcmin = {self.uK_arcmin})')
        print(f'    red_ampl = {self.red_ampl}    (l_knee = {self.l_knee})')


    def _init_white_noise(self, weight_map, ivar=None):
        """Helper method called by constructor.
        
        self.w2 = sum(weight_map^2 * pixsize) / (4pi)
                = approximate normalization used to compute pseudo-Cls

        self.w2_ivar = sum(weight_map^2 * pixsize^2 / ivar) / (4pi)

        self.ivar_uK_arcmin = sqrt(w2_ivar / w2) * (60*180/pi)
                            = equivalent noise level of ivar map
        """

        wp = weight_map * weight_map.pixsizemap()
        self.w2 = np.vdot(weight_map, wp) / (4*np.pi)
        
        if ivar is None:
            self.w2_ivar = None
            return
        
        invalid = (ivar <= 0)
        
        if np.any(weight_map * invalid):
            raise RuntimeError("CmbClFitter: weight map is nonzero in pixel with ivar <= 0")
        
        wpi = wp / np.where(invalid, 1.0, ivar)   # weight_map * pixsize / ivar
        w2_ivar = np.vdot(wp, wpi) / (4*np.pi)
        self.ivar_uK_arcmin = np.sqrt(w2_ivar/self.w2) * 60*180/np.pi
        print(f'CmbClFitter: equivalent noise level of ivar map = {self.ivar_uK_arcmin} uK-arcmin')
        

    def _cl_red_noise(self, *, alpha=None, lred=None, lmax=None):
        if alpha is None:
            alpha = self.alpha
        if lred is None:
            lred = self.lred
        if lmax is None:
            lmax = self.lmax

        cl = np.maximum(np.arange(lmax+1)/lred, 1.0)**(alpha)
        
        if self.beamed_red_noise:
            cl *= self.bl[:(lmax+1)]**2

        return cl


    def _pcl_from_weighted_map(self, wm):
        alm = pixell_utils.map2alm(wm, self.lmax)
        cl = pixell.curvedsky.alm2cl(alm)
        return cl / self.w2


    def _pcl_from_cl(self, cl_true, weight_map):
        """Simulate map with power spectrum 'cl_true', apply weight map, return (pseudo) Cls.
        
        Some day, I'll implement the pseudo-Cl transfer matrix, and this simulation-based
        approach won't be needed."""

        assert cl_true.shape == ((self.lmax + self.lpad + 1),)
        alm = pixell.curvedsky.rand_alm(cl_true)
        m = pixell_utils.alm2map(alm, weight_map.shape, weight_map.wcs)
        m *= weight_map
        return self._pcl_from_weighted_map(m)

        
    def _plot_smoothed(self, y, s, label=None, l2=False):
        """Helper method for plot()."""
        
        assert y.shape == (self.lmax+1,)
        l = np.arange(self.lmax+1)
        y = (y*l*(l+1)/(2*np.pi)) if l2 else y
        l_smoothed = utils.boxcar_sum(l[self.lmin:], s, normalize=True)
        y_smoothed = utils.boxcar_sum(y[self.lmin:], s, normalize=True)
        plt.plot(l_smoothed, y_smoothed, label=label)
