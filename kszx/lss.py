import numpy as np

from . import Box
from . import cpp_kernels
from . import utils
    

####################################################################################################


def fft_r2c(box, arr):
    """Computes the FFT of real-space map 'arr', and returns a Fourier-space map.

    Args:
        box: instance of class kszx.Box.
        arr: numpy array with shape (box.real_space_shape) and real dtype.

    Returns:
        numpy array with shape (box.fourier_space_shape) and complex dtype.

    Reminder: our Fourier conventions are

        f(k) = (pixel volume) sum_x f(x) e^{-ik.x}     [ morally int d^nx f(x) e^{-ik.x} ]
        f(x) = (box volume)^{-1} sum_k f(k) e^{ik.x}   [ morally int d^nk/(2pi)^n f(k) e^{ik.x} ]        
    """

    assert isinstance(box, Box)
    assert box.is_real_space_map(arr)   # check shape and dtype of input array
    
    ret = np.fft.rfftn(arr)
    ret *= box.pixel_volume   # see Fourier conventions in docstring
    return ret                # numpy array with shape=box.fourier_space_shape and dtype=complex.


def fft_c2r(box, arr):
    """Computes the FFT of Fourier-space map 'arr', and returns a real-space map.

    Args:
        box: instance of class kszx.Box.
        arr: numpy array with shape (box.fourier_space_shape) and complex dtype.

    Returns:
        numpy array with shape (box.real_space_shape) and real dtype.

    Reminder: our Fourier conventions are

        f(k) = (pixel volume) sum_x f(x) e^{-ik.x}     [ morally int d^nx f(x) e^{-ik.x} ]
        f(x) = (box volume)^{-1} sum_k f(k) e^{ik.x}   [ morally int d^nk/(2pi)^n f(k) e^{ik.x} ]        
    """
    
    assert isinstance(box, Box)
    assert box.is_fourier_space_map(arr)   # check shape and dtype of input array
    
    ret = np.fft.irfftn(arr, box.npix)
    ret *= (1.0 / box.pixel_volume)    # see Fourier conventions in docstring
    return ret                         # numpy array with shape=box.real_space shape and dtype=complex.


####################################################################################################


def interpolate_points(box, grid, points, kernel):
    """Interpolates real-space grid at a specified set of points.

    Args:
        box: instance of class kszx.Box.
        grid: numpy array with shape (box.real_space_shape) and real dtype.
        points: numpy array with shape (npoints, box.ndim)
        kernel: currently only 'cic' is supported.

    Returns:
        1-d numpy array with length npoints.
 
    Note: the 'points' array contains coordinates in a coordinate system where the observer is
    at the origin. That is, the translation between the 'points' array and pixel indices is:
    
        (pixel indices) = (points - box.lpos) / box.pixsize.
    """

    assert isinstance(box, Box)
    assert box.is_real_space_map(grid)   # check grid shape, dtype
    assert points.ndim == 2
    assert points.shape[1] == box.ndim

    if box.ndim != 3:
        raise RuntimeError('kszx.interpolate_points(): currently only ndim==3 is supported')
    if (kernel != 'cic') and (kernel != 'CIC'):
        raise RuntimeError('kszx.interpolate_points(): currently only kernel=="cic" is supported')

    return cpp_kernels.cic_interpolate_3d(grid, points, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)


def grid_points(box, grid, points, kernel, weights=None):
    """Add a sum of delta functions, with specified coefficients or 'weights', to a real-space map.

    Args:
        box: instance of class kszx.Box.
        grid: numpy array with shape (box.real_space_shape) and real dtype.
        points: numpy array with shape (npoints, box.ndim)
        kernel: currently only 'cic' is supported.
        weights: either scalar, None, or 1-d array with length npoints.
           - if 'weights' is a scalar, then all delta functions have equal weight.
           - if 'weights' is None, then all delta functions have weight 1.
    
    Returns: None.

    A note on normalization: 

    The normalization of the output map includes a factor (1 / pixel volume). (That is, the sum of
    the output array is np.sum(weights) / box.pixel_volume, not np.sum(weights).) This normalization 
    best represents a weighted sum of delta functions f(x) = sum_j w_j delta^3(x-x_j). For example:

       - If we integrate the output array over volume:
           integral = np.sum(grid) * box.pixel_volume
         then we get np.sum(weights), as expected for a sum of delta functions sum_j w_j delta^3(x-x_j).

       - If we FFT the output array with ksz.lss.fft_r2c(), the result is a weighted sum of plane waves:
           sum_j w_j exp(-ik.x_j)   
         with no factor of box or pixel volume.

    More notes:

      - The weighted sum of delta functions will be added to the current contents of 'grid'. 
        (If this is not desired, you should zero the 'grid' array before calling grid_points().)
    
      - The 'points' array contains coordinates in a coordinate system where the observer is
        at the origin. That is, the translation between the 'points' array and pixel indices is:
    
             (pixel indices) = (points - box.lpos) / box.pixsize.
    """
    
    assert isinstance(box, Box)
    assert box.is_real_space_map(grid)   # check grid shape, dtype
    assert points.ndim == 2
    assert points.shape[1] == box.ndim
    npoints = points.shape[0]

    if weights is None:
        weights = np.ones(npoints)

    weights = np.asarray(weights, dtype=float)

    if weights.ndim == 0:
        weights = np.full(npoints, fill_value=float(weights))
    if weights.shape != (npoints,):
        raise RuntimeError("kszx.grid_points(): 'points' and 'weights' arrays don't have consistent shapes")
        
    if box.ndim != 3:
        raise RuntimeError('kszx.grid_points(): currently only ndim==3 is supported')
    if (kernel != 'cic') and (kernel != 'CIC'):
        raise RuntimeError('kszx.grid_points(): currently only kernel=="cic" is supported')
    
    cpp_kernels.cic_grid_3d(grid, points, weights, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)


####################################################################################################


def _multiply(src, x, dest, in_place):
    """Helper for functions which take 'dest' and 'in_place' arguments.
        (multiply_rfunc(), multiply_kfunc(), multiply_r_component(), apply_partial_derivative()."""
    
    if (dest is not None) and in_place:
        raise RuntimeError("Specifying both 'dest' and 'in_place' arguments is not allowed")
    if (dest is not None) and (dest.shape != src.shape):
        raise RuntimeError("'dest' array has wrong shape")
    if (dest is not None) and (dest.shape != src.dtype):
        raise RuntimeError("'dest' array has wrong dtype")

    if (dest is src) or in_place:
        src *= x
        return src
    elif (dest is not None):
        dest[:] = src[:]
        dest *= x
        return dest
    else:
        return src * x


def multiply_rfunc(box, arr, f, dest=None, in_place=False, regulate=False, eps=1.0e-6):
    """Multiply real-space map 'arr' by a function f(r), where r is scalar radial coordinate.

    Args:
        box: instance of class kszx.Box.
        arr: numpy array representing a real-space map (i.e. shape=box.real_space_shape, dtype=float)
        f: function (or callable object) representing the function r -> f(r).
        dest: real-space map where output will be written (if None, then new array will be allocated)
        in_place: setting this to True is equivalent to dest=arr.
        regulate (boolean): if True, then replace r = max(r,eps*pixsize) before calling f().
        eps (float): only used if regulate=True.

    Returns: None. (The real-space map 'arr' is modified in-place.)

    Notes: 
    
       - The function f() must be vectorized: its argument 'r' will be a 3-dimensional arary,
         and the return value should be an array with the same shape.

       - r-values passed to f() will be in "observer" coordinates, i.e. the observer is at
         the origin, and the box corners are given by self.lpos, self.rpos.
    """

    assert isinstance(box, Box)
    assert box.is_real_space_map(arr)   # check shape, dtype
    assert callable(f)
    assert eps < 0.5

    r = box.get_r(regulate=regulate, eps=eps)
    fr = f(r)
    
    if not box.is_real_space_map(fr):
        raise RuntimeError('kszx.lss.multiply_rfunc(): function f(r) returned unexpected shape/dtype')

    return _multiply(arr, fr, dest, in_place)
    
    
def multiply_kfunc(box, arr, f, dest=None, in_place=False, dc=None):
    """Multiply Fourier-space map 'arr' in-place by a real-valued function f(k), where k=|k| is scalar wavenumber.

    Args:
        box: instance of class kszx.Box.
        arr: numpy array representing a Fourier-space map (i.e. shape=box.fourier_space_shape, dtype=complex)
        f: function (or callable object) representing the function k -> f(k).
        dest: real-space map where output will be written (if None, then new array will be allocated)
        in_place: setting this to True is equivalent to dest=arr.
        dc (float): if True, then f() is not evaluated at k=0, and the value of 'dc' is used instead of f(0).

    Returns: None. (The Fourier-space map 'arr' is modified in-place.)

    Notes: 
    
       - The function f() must be vectorized: its argument 'k' will be a 3-dimensional arary,
         and the return value should be an array with the same shape.

       - k-values passed to f() will include the factor (2pi / boxsize).

       - If 'dc' is specified, then f() will not be evaluated at k=0.
         For example, this code is okay (will not raise a divide-by-zero expection):

           multiply_kfunc(box, arr, lambda k:1/k**2, dc=0.0)
    """

    assert isinstance(box, Box)
    assert box.is_fourier_space_map(arr)   # check shape, dtype
    assert callable(f)

    k = box.get_k(regulate = (dc is not None))
    fk = f(k)

    if fk.shape != box.fourier_space_shape:
        raise RuntimeError('kszx.lss.multiply_kfunc(): function f(k) returned unexpected shape')
    if fk.dtype != float:
        raise RuntimeError('kszx.lss.multiply_kfunc(): function f(k) returned dtype={fk.dtype} (expected float)')

    if dc is not None:
        fk[(0,)*box.ndim] = dc

    return _multiply(arr, fk, dest, in_place)


def multiply_r_component(box, arr, axis, dest=None, in_place=True):
    """Multiply real-space map 'arr' in-place by r_j (the j-th Cartesian coordinate, in observer coordinates).

    Args:
        box: instance of class kszx.Box.
        arr: numpy array representing a real-space map (i.e. shape=box.real_space_shape, dtype=float)
        dest: real-space map where output will be written (if None, then new array will be allocated)
        in_place: setting this to True is equivalent to dest=arr.
        axis (integer): component 0 <= j < box.ndim.

    Returns: None. (The real-space map 'arr' is modified in-place.)

    Note: 

       - Values of r_i will be signed, and in "observer" coordinates, i.e. the observer is at
         the origin, and the box corners are given by self.lpos, self.rpos.
    """
    
    assert isinstance(box, Box)
    assert box.is_real_space_map(arr)

    ri = box.get_r_component(axis)
    return _multiply(arr, ri, dest, in_place)

    
def apply_partial_derivative(box, arr, axis, dest=None, in_place=True):
    """Multiply Fourier-space map 'arr' in-place by (i k_j). (This is the partial derivative d_j in Fourier space.)

    Args:
        box: instance of class kszx.Box.
        arr: numpy array representing a Fourier-space map (i.e. shape=box.fourier_space_shape, dtype=complex)
        dest: real-space map where output will be written (if None, then new array will be allocated)
        in_place: setting this to True is equivalent to dest=arr.
        axis (integer): component 0 <= j < box.ndim.

    Returns: None. (The Fourier-space map 'arr' is modified in-place.)

    Notes: 
    
       - Values of k_j will be signed, and include the factor (2pi / boxsize).

       - The value of k_j will be taken to be zero at the Nyquist frequency.
         (I think this is the only sensible choice, since the sign is ambiguous.)
    """

    assert isinstance(box, Box)
    assert box.is_fourier_space_map(arr)

    ki = 1j * box.get_k_component(axis, zero_nyquist=True)
    return _multiply(arr, ki, dest, in_place)


####################################################################################################


def _to_float(x, errmsg):
    try:
        return float(x)
    except:
        raise RuntimeError(errmsg)


def _sqrt_pk(box, pk, regulate):
    """Helper for simulate_gaussian_field()."""

    if callable(pk):
        k = box.get_k(regulate=regulate)
        pk = pk(k)
        
        if pk.shape != box.fourier_space_shape:
            raise RuntimeError('kszx.lss.simulate_gaussian_field(): function pk() returned unexpected shape')
        if pk.dtype != float:
            raise RuntimeError('kszx.lss.simulate_gaussian_field(): function pk() returned dtype={pk.dtype} (expected float)')
        if np.min(pk) < 0:
            raise RuntimeError('kszx.lss.simulate_gaussian_field(): function pk() returned negative values')

        del k
        pk **= 0.5
        return pk   # returns sqrt(P(k))

    pk = _to_float(pk, 'kszx.lss.simulate_gaussian_field(): expected pk argument to be either callable, or a real scalar')

    if pk < 0:
        raise RuntimeError('kszx.lss.simulate_gaussian_field(): expected scalar pk argument to be non-negative')
    
    return np.sqrt(pk)

    
def simulate_white_noise(box, *, fourier):
    """Simulate white noise, in either real space or Fourier space, normalized to P(k)=1.

    Args:

        box: instance of class kszx.Box.
        fourier (boolean): determines whether output is real-space or Fourier-space.
    
    Returns: numpy array

       - if fourier=False, real-space map is returned 
            (shape = box.real_space_shape, dtype = float)

       - if fourier=True, Fourier-space map is returned 
            (shape = box.fourier_space_shape, dtype = complex)

    Intended as a helper for simulate_gaussian_field(), but may be useful on its own.

    Reminder: our Fourier conventions imply
    
        <f(k) f(k')^*> = (box volume) P(k) delta_{kk'}  [ morally P(k) (2pi)^n delta^n(k-k') ]

    Note units.
    """

    if not fourier:
        rms = 1.0 / np.sqrt(box.pixel_volume)
        return np.random.normal(size=box.real_space_shape, scale=rms)
        
    # Simulate white noise in Fourier space.
    nd = box.ndim
    rms = np.sqrt(0.5 * box.box_volume)
    ret = np.zeros(box.fourier_space_shape, dtype=complex)        
    ret.real = np.random.normal(size=box.fourier_space_shape, scale=rms)
    ret.imag = np.random.normal(size=box.fourier_space_shape, scale=rms)

    # The rest of this function imposes the reality condition f(-k) = f(k)^*.
    
    # t = modes where k_{nd-1} is self-conjugate
    n = box.npix[nd-1]
    s1 = (slice(None),) * (nd-1)
    s2 = slice(0,1) if (n % 2) else slice(0, (n//2)+1, (n//2))
    tview = ret[s1+(s2,)] 
    tcopy = np.conj(tview)   # copy and complex conjugate

    # Apply parity operation k -> (-k) to 'tcopy'.
    for axis in range(nd-1):
        n = box.npix[axis]
        s1 = (slice(None),) * axis
        s2fwd = (slice(1,n),)
        s2rev = (slice(n-1,0,-1),)
        s3 = (slice(None),) * (nd-axis-1)
        u = np.copy(tcopy[s1+s2rev+s3])
        tcopy[s1+s2fwd+s3] = u

    # Replace f(k) by (f(k) - f(-k)^*) / sqrt(2)
    tview += tcopy
    tview *= np.sqrt(0.5)   # preserve variance
    return ret


def simulate_gaussian_field(box, pk, pk0=None):
    """
    Args:
        pk (either callable, or a scalar). Must be real-valued.
        pk0 (either scalar, or None)

    Note division by zero.
    Note units.

    Reminder: our Fourier conventions imply
    
        <f(k) f(k')^*> = (box volume) P(k) delta_{kk'}  [ morally P(k) (2pi)^n delta^n(k-k') ]
    """

    assert isinstance(box, Box)

    sqrt_pk = _sqrt_pk(box, pk, regulate = (pk0 is not None))
    ret = simulate_white_noise(box, fourier=True)

    dc = ret[(0,)*box.ndim]   # must precede multiplying by sqrt_pk
    ret *= sqrt_pk

    if pk0 is not None:
        pk0 = _to_float(pk0, 'kszx.lss.simulate_gaussian_field(): expected pk0 argument to be a real scalar')
        if pk0 < 0:
            raise RuntimeError('kszx.lss.simulate_gaussian_field(): expected pk0 argument to be non-negative')
        ret[(0,)*box.ndim] = np.sqrt(pk0) * dc
    
    return ret


####################################################################################################


def _parse_map_or_maps(box, map_or_maps):
    """Helper for estimate_power_spectrum().
    Returns (map_list, multi_map_flag)."""

    if box.is_fourier_space_map(map_or_maps):
        return ([map_or_maps], False)  # single map

    try:
        map_list = list(map_or_maps)
    except:
        return ([], False)

    for x in map_list:
        if not box.is_fourier_space_map(x):
            return ([], False)

    return (map_list, True)
    

def estimate_power_spectrum(box, map_or_maps, kbin_delim, *, use_dc=False, allow_empty_bins=False, return_counts=False):
    """
    Args:

        box: instance of class kszx.Box.

        map_or_maps: single or multiple Fourier-space maps
          - single map: numpy array of shape (box.fourier_space_shape) and dtype=complex.
          - multiple maps: iterable object returning numpy arrays of shape (box.fourier_space_shape)
              and dtype=complex. (Could be a list, tuple, or a numpy array with an extra axis.)

        kbin_delim: 1-d array of length (nkbins+1)

    Returns: 

       - if return_counts=False (the default), then return value is an array 'pk'.
            - if input is a single map, then pk.shape = (nkbins,)
            - if input is multiple maps, then pk.shape = (nmaps, nmaps, nkbins)

       - if return_counts=True, then return value is a pair (pk, bin_counts).

    Notes:

        - Note on kbin_delim[0] 

        - Reminder: our Fourier conventions imply
            <f(k) f(k')^*> = (box volume) P(k) delta_{kk'}
    """

    kbin_delim = np.asarray(kbin_delim, dtype=float)
    
    assert kbin_delim.ndim == 1
    assert len(kbin_delim) >= 2    
    assert isinstance(box, Box)

    map_list, multi_map_flag = _parse_map_or_maps(box, map_or_maps)
    
    if len(map_list) == 0:
        raise RuntimeError("kszx.lss.estimate_power_spectrum(): expected 'map_or_maps' arg to be either"
                           + "a Fourier-space map, or an iterable returning Fourier-space maps")

    
    if len(map_list) > 4:
        # Note: If the C++ code is modified to allow larger numbers of maps,,
        # make sure to update the unit test too (test_estimate_power_spectrum()).
        raise RuntimeError("kszx.lss.estimate_power_spectrum(): we currently only support nmaps <= 4."
                           + " This is a temporary problem that I'll fix later. It needs minor changes"
                           + " to the C++ code.")

    
    assert kbin_delim[0] >= 0.
    assert utils.is_sorted(kbin_delim)
    
    if (not use_dc) and (kbin_delim[0] == 0):
        kbin_delim = np.copy(kbin_delim)
        kbin_delim[0] = min(np.min(box.kfund), kbin_delim[1]) / 2.
    
    pk, bin_counts = cpp_kernels.estimate_power_spectrum(map_list, kbin_delim, box.npix, box.kfund, box.box_volume)

    if (not allow_empty_bins) and (np.min(bin_counts) == 0):
        raise RuntimeError('kszx.lss.estimate_power_spectrum(): some k-bins were empty')
    
    if not multi_map_flag:
        pk = pk[0,0,:]   # shape (1,1,nkbins) -> shape (nkbins,)

    return (pk, bin_counts) if return_counts else pk
