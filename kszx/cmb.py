import healpy
import pixell.enmap
import numpy as np

from . import utils


def fkp_from_ivar(ivar, cl0, normalize=True, return_wvar=False):
    """
    The 'ivar' argument is an inverse noise varaiance map (i.e. pixell.enmap) in (uK)^{-2},
    for example the return value of act.DataRelease.read_ivar().

    The 'cl0' argument parameterizes the FKP weight function.
      - cl0=0 corresponds to inverse noise weighting.
      - Large cl0 corresponds to uniform weighing (but cl0=inf won't work).
      - Intuitively, cl0 = "fiducial signal C_l at wavenumber l of interest".
      - I usually use cl0 = 0.01 for plotting all-sky CMB temperature maps.
      - I usually use cl0 ~ 10^(-5) for small-scale KSZ filtering.

    The FKP weight function is defined by

      W(x) = 1 / (Cl0 + Nl(x))     if normalize=False

    where Nl(x) = pixarea(x) / ivar(x) is the "local" noise power spectrum. In implementation, 
    in order to avoid divide-by-zero for Cl0=0 or ivar=0, we compute W(x) equivalently as:

      W(x) = ivar(x) / (pixarea(x) + Cl0 * ivar(x))         if normalize=False

    If normalize=True, then we normalize the weight function so that max(W)=1.
    
    If wvar=True, then we return W(x) var(x) = W(x) / ivar(x), instead of returning W(x).
    """
    
    assert isinstance(ivar, pixell.enmap.ndmap)
    assert ivar.ndim == 2
    assert np.all(ivar >= 0.0)
    assert cl0 >= 0.0

    wvar = 1.0 / (ivar.pixsizemap() + cl0 * ivar)
    w = wvar * ivar
    
    wmax = np.max(w)
    assert wmax > 0.0

    ret = wvar if return_wvar else w
    
    if normalize:
        ret /= wmax

    return ret


def estimate_cl(alm_or_alms, lbin_delim):
    """Similar interface to lss.estimate_power_spectrum().

    It's okay if lbin_delim is a float array (it will be converted to int).

    If 'alm_or_alms' is a 1-d array (single alm), returns a 1-d array of length (nlbins,).
    If 'alm_or_alms' is a 2-d array (multiple alms), returns a 3-d array of shape (nalm,nalm,nlbins).
    """

    alm_or_alms = utils.asarray(alm_or_alms, 'kszx.estimate_cl()', 'alm_or_alms', dtype=complex)    
    lbin_delim = utils.asarray(lbin_delim, 'kszx.estimate_cl()', 'lbin_delim', dtype=int)
    multi_flag = True

    # Check 'lbin_delim' arg.
    if lbin_delim.ndim != 1:
        raise RuntimeError("kszx.estimate_cl(): expected 'lbin_delim' to be a 1-d array")
    if len(lbin_delim) < 2:
        raise RuntimeError("kszx.estimate_cl(): expected 'lbin_delim' to have length >= 2")
    if not utils.is_sorted(lbin_delim):
        raise RuntimeError("kszx.estimate_cl(): 'lbin_delim' was not sorted, or bins were too narrow")
    if lbin_delim[0] < 0:
        raise RuntimeError("kszx.estimate_cl(): expected 'lbin_delim' elements to be >= 0")
    
    # Convert to 2-d.
    if alm_or_alms.ndim == 1:
        alm_or_alms = np.reshape(alm_or_alms, (1,-1))
        multi_flag = False
    elif alm_or_alms.ndim != 2:
        raise RuntimeError("kszx.estimate_cl(): 'alm_or_alms' array did not have expected shape")

    nalm, nlm = alm_or_alms.shape
    lmax = int(np.sqrt(2*nlm) - 1)

    if (nlm == 0) or ((2*nlm) != (lmax+1)*(lmax+2)):
        raise RuntimeError("kszx.estimate_cl(): 'alm_or_alms' array did not have expected shape")
    if (lbin_delim[-1] > lmax):
        raise RuntimeError(f"kszx.estimate_cl(): l-bin endpoint (={lbin_delim[-1]}) was > alm_lmax (={lmax})")
    
    nlbins = len(lbin_delim) - 1
    cl = np.zeros((nalm,nalm,lmax+1))
    ret = np.zeros((nalm,nalm,nlbins))
    
    # FIXME I think this can be done with one call to pixell.alm2cl().
    for i in range(nalm):
        for j in range(i+1):
            cl[i,j,:] = cl[j,i,:] = healpy.alm2cl(alm_or_alms[i,:], alm_or_alms[j,:])

    for b in range(nlbins):
        bin_lmin, bin_lmax = lbin_delim[b], lbin_delim[b+1]
        ret[:,:,b] = np.mean(cl[:,:,bin_lmin:bin_lmax], axis=2)

    if not multi_flag:
        ret = ret[0,0,:]

    return ret

        
    
