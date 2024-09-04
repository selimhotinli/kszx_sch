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
 
    Note: the 'points' array contains coordinates in a coordinate system where the observer
    is at the origin. That is, the translation between the 'points' array and pixel indices is:
    
        (pixel indices x,y,z) = (points - box.lpos) / box.pixsize.
    """

    points = np.asarray(points)
    
    assert isinstance(box, Box)
    assert box.is_real_space_map(grid)
    assert points.shape[-1] == box.ndim

    if box.ndim != 3:
        raise RuntimeError('kszx.interpolate(): currently only ndim==3 is supported')
    if (kernel != 'cic') and (kernel != 'CIC'):
        raise RuntimeError('kszx.interpolate(): currently only kernel=="cic" is supported')

    return cpp_kernels.cic_interpolate_3d(grid, points, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)


