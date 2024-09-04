import numpy as np

from . import Box
from . import cpp_kernels


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
