import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from . import utils

from .Catalog import Catalog
from .Cosmology import Cosmology
from .KszPipe import KszPipeOutdir


class PgvLikelihood:
    def __init__(self, data, surrs, bias_matrix, jeffreys_prior=False):
        r"""Likelihood function for one or more kSZ power spectra of the form $P_{gv}(k)$.

        This class can be used to fit $(f_{NL}, b_v)$ values or run MCMCs.
        Currently limited to using $P_{gv}(k)$ only -- it would be interesting to
        extend the analysis to include $P_{gg}(k)$ and/or $P_{vv}(k)$.
        
        The PgvLikelihood constructor has an abstract syntax, which may not be the
        most convenient. Instead of calling the constructor directly, you may want
        to call :meth:`~kszx.PgvLikelihood.from_pipeline_outdir()` as follows::

           dirname = ...   # directory name containing pipeline outputs
           pout = kszx.KszPipeOutdir(dirname)

           # The 'fields=[[1,0]]' argument selects 90 GHz.
           # See the from_pipeline_outdir() docstring for more examples.
           lik = kszx.PgvLikelihood.from_pipeline_outdir(pout, fields=[[1,0]], nkbins=10)

        The rest of this docstring describes the PgvLikelihood constructor syntax
        (even though, as explained above, you probably don't want to call the constructor
        directly!) First we define:

          - B = number of bias parameters in the MCMC (usually 1)
          - V = number of velocity reconstructions (usually 1)
          - K = number of k-bins
          - S = number of surrogate sims
          - D = V*K = number of components in "data vector".

        Note the distinction between a 90+150 GHz "sum" fit (V=1), and a 90+150 GHz
        "joint" fit (V=2). In the first case, we add the 90+150 bandpowers and construct
        a likelihood based on their sum, and in the second case we do a joint fit to
        both sets of bandpowers (i.e. doubling the size of the data vector and
        covariance matrices).
        
        Then the constructor arguments are as follows:

          - ``data``: $P_{gv}(k)$ "data" bandpowers, an array of shape ``(V,K)``.
        
          - ``surrs``: $P_{gv}(k)$ surrogate bandpowers, an array of shape ``(S,2,2,V,K)``.
            The first length-2 index is an fnl exponent in {0,1}.
            The second length-2 index is a bv exponent in {0,1}.

          - ``bias_matrix``: array of shape $(B,V)$, where $B \le V$. This gives the
            correspondence between bias parameters and velocity reconstruction.
            There are basically 3 cases of interest here:

              - ``bias_matrix = [[1,1]]``: analysis of single velocity reconstruction field (V=1).
        
              - ``bias_matrix = [[1,1]]``: joint analysis of two velocity reconstruction fields (V=2),
                with bias assumed to be the same for both freqs.

              - ``bias_matrix = [[1,0],[0,1]]``: joint analysis of two velocity reconstruction
                fields (V=2), with two independent bias parameters.
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
        r"""Constructs a PgvLikelhood from a :class:`~kszx.KszPipeOutdir` instance.

        Usually more convenient than calling the PgvLikelihood constructor directly.
        Example usage::
        
           dirname = ...   # directory name containing pipeline outputs
           pout = kszx.KszPipeOutdir(dirname)

           # The 'fields=[[1,0]]' argument selects 90 GHz.
           # See below for more examples.
           lik = kszx.PgvLikelihood.from_pipeline_outdir(pout, fields=[[1,0]], nkbins=10)

        The ``pout`` argument is a :class:`~kszx.KszPipe.KszPipeOutdir` object.
        
        The ``fields`` argument is a V-by-2 matrix. Each row of the matrix selects a linear
        combination of the 90+150 GHz velocity reconstructions. For example:
        
           - ``fields = [[1,0]]`` for 90 GHz analysis
           - ``fields = [[0,1]]`` for 150 GHz analysis
           - ``fields = [[1,1]]`` for (90+150) "sum map" analysis
           - ``fields = [[1,-1]]`` for null (90-150) "null map" analysis
           - ``fields = [[1,0],[0,1]]`` for joint analysis with both (90+150 GHz) sets of bandpowers

        The ``multi_bias`` argument is only needed for a joint analysis (i.e. $V > 1$) and determines
        whether each freq channel has an independent bias parameter (``multi_bias=True``), or whether
        all frequency channels have the same bias (``multi_bias=False``).
        """
        
        assert isinstance(pout, KszPipeOutdir)
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
        r"""Returns a shape ``(S,V,K)`` array, by "specializing" surrogates to specified $(f_{NL}, b_v)$.
        
        The ``bv`` arugment should be an array of shape ``(B,)`` where $B$ is the number
        of bias parameters in the likelihood. If $B=1$, then ``bv`` can be a scalar.

        Convenient but slow! Used in many places, but not :meth:`~kszx.PgvLikelihood.fast_likelihood()`.
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
        r"""Computes mean and covaraince of surrogates, for specified $(f_{NL}, b_v)$.

        Returns ``(mean, cov)`` where ``mean.shape=(D,)`` and ``cov.shape=(D,D).``

        The ``bv`` arugment should be an array of shape ``(B,)`` where $B$ is the number
        of bias parameters in the likelihood. If $B=1$, then ``bv`` can be a scalar.
        
        Brute force implementation: we call :meth:`kszx.PgvLikelihood.specialize_surrogates()``,
        then compute the mean and covariance with ``np.mean()`` and ``np.cov()``.
        """
        
        s = self.specialize_surrogates(fnl, bv, flatten=True)  # shape (S,D)
        mean = np.mean(s, axis=0)       # shape (D,)
        cov = np.cov(s, rowvar=False)   # shape (D, D)
        return mean, cov

    
    def slow_mean_and_cov_gradients(self, fnl, bv):
        r"""Computes the gradient of the mean and covariance with respect to $(f_{NL}, b_v)$.

        Returns ``grad_mean, grad_cov``, where both gradients are represented as arrays
        with an extra length-(B+1) axis, i.e.

          - ``mean.shape = (D,)  =>  grad_mean.shape = (B+1,D)``
          - ``cov.shape = (D,D)  =>  grad_cov.shape = (B+1,D,D)``
          
        Uses boneheaded algorithm: since mean, cov are at most quadratic in $f_{NL}$ and $b_v$,
        naive finite difference is exact (and independent of step sizes).

        The ``bv`` arugment should be an array of shape ``(B,)`` where $B$ is the number
        of bias parameters in the likelihood. If $B=1$, then ``bv`` can be a scalar.

        This method is only used to compute log-likelihoods with the Jeffreys prior.
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
        r"""Returns the log-likelihood at the specified $(f_{NL}, b_v)$.

        The ``bv`` arugment should be an array of shape ``(B,)`` where $B$ is the number
        of bias parameters in the likelihood. If $B=1$, then ``bv`` can be a scalar.
        """
        
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
        r"""Initializes ``self.samples`` to an array of shape (N,B+1) where N is large."""

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
        r"""Makes a corner plot from MCMC results. Intended to be called from jupyter."""

        if not hasattr(self, 'samples'):
            raise RuntimeError('Must call PgvLikelihood.run_mcmc() before PgvLikelihood.show_mcmc().')
        
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
        r"""Computes a $\chi^2$ statistic, which compares $P_{gv}^{data}(k)$ to model with given $(f_{NL}, b_v)$.

        Returns ($\chi^2$, $N_{dof}$, $p$-value).

        The ``bv`` arugment should be an array of shape ``(B,)`` where $B$ is the number
        of bias parameters in the likelihood. If $B=1$, then ``bv`` can be a scalar.
        
        The ``ddof`` argument is used to compute the number of degrees of freedom, as
        ``ndof = nkbins - ddof``. If ``ddof=None``, then it will be equal to the
        number of nonzero (fnl, bias) params. (This is usually the correct choice.)
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
        r"""Returns $(f_{NL}, b_v)$ obtained by maximizing joint likelihood.
        
        Note that $b_v$ is represented as a 1-d array with shape ``(B,)``, even if $B=1$.
        """

        x0 = np.zeros(self.B+1)
        x0[0] = fnl0
        x0[1:] = bv0

        f = lambda x: -self.fast_log_likelihood(x[0], x[1:])  # note minus sign
        result = scipy.optimize.minimize(f, x0, method='Nelder-Mead')
        
        fnl = result.x[0]
        bv = result.x[1:]
        return fnl, bv        

        
    def fit_bv(self, fnl=0, bv0=0.3):
        r"""Returns $b_v$ obtained by maximizing conditional likelihood at the given $f_{NL}.
        
        Note that $b_v$ is represented as a 1-d array with shape ``(B,)``, even if $B=1$.
        """

        x0 = np.full(self.B, bv0)
        f = lambda x: -self.fast_log_likelihood(fnl, x)  # note minus sign
        result = scipy.optimize.minimize(f, x0, method='Nelder-Mead')
        
        bv = result.x
        return bv


    def compute_snr(self):
        r"""Returns total SNR of $P_{gv}(k)$. Does not assume a model for $P_{gv}(k)$."""
        
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
        r"""Equivalent to :meth:`~kszx.PgvLikelihood.slow_mean_and_cov()`, but faster. Intended for use in MCMC.
        
        Note that bv must be a 1-d array of length B, i.e. scalar is not allowed."""
        
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
        r"""Equivalent to :meth:`~kszx.PgvLikelihood.slow_log_likelihood()`, but faster. Intended for use in MCMC.
        
        Note that ``bv`` must be a 1-d array of length B, i.e. scalar is not allowed."""
        
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
        r"""Test fast_mean_and_cov(), by checking that it agrees with slow_mean_and_cov() at 10 random points."""

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
        r"""Test fast_log_likelihood(), by checking that it agrees with slow_log_likelihood() at 10 random points."""
        
        for _ in range(10):
            fnl = np.random.uniform(-50, 50)
            bv = np.random.uniform(0, 1, size=(self.B,))
            logL_slow = self.slow_log_likelihood(fnl, bv)
            logL_fast = self.fast_log_likelihood(fnl, bv)
            assert np.abs(logL_slow - logL_fast) < 1.0e-10
        
    
    @staticmethod
    def make_random():
        r"""Construct and return a PgvLikelihood with random (data, surrs, bias_matrix).
        
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
