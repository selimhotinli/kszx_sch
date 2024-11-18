import os
import healpy
import scipy.fft
import pixell.enmap
import numpy as np

from .Box import Box
from . import cpp_kernels
from . import utils
    

####################################################################################################


def fft_r2c(box, arr, spin=0, threads=None):
    r"""Computes the FFT of real-space map 'arr', and returns a Fourier-space map.

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``arr`` (array): numpy array representing a real-space map (dtype=float).

        - ``spin`` (integer): currently only spin=0 and spin=1 are supported.

        - ``threads`` (integer or None): number of parallel threads used.
          If ``threads=None``, then number of threads defaults to ``os.cpu_count()``.

    Returns a numpy array representing a Fourier-space map (dtype=complex).

    The real-space and Fourier-space array shapes are given by ``box.real_space_shape``
    and ``box.fourier_space_shape``, and are related as follows:

    $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
    $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

    Notes:

       - Our spin-0 Fourier conventions are:

          $$f(k) = V_{pix} \sum_x f(x) e^{-ik\cdot x}$$

          $$f(x) = V_{box}^{-1} \sum_k f(k) e^{ik\cdot x}$$

       - We define spin-1 Fourier transforms by inserting an extra factor
         $(\pm i {\hat k} \cdot {\hat r})$:

          $$f(k) = V_{pix} \sum_x f(x) (-i {\hat k} \cdot {\hat r}) e^{-ik\cdot x}$$

          $$f(x) = V_{box}^{-1} \sum_k f(k) (i {\hat k} \cdot {\hat r}) e^{ik\cdot x}$$

         where the line-of-sight direction $\hat r$ is defined in "observer coordinates"
         (see :class:`~kszx.Box` for more info).

         Application: the spin-1 c2r transform can be used to compute the radial velocity
         field from the density field (with a factor $faH/k$).

         Another application: the spin-1 r2c transform can be used to estimate $P_{gv}(k)$
         or $P_{vv}(k)$ from the radial velocity field (or the kSZ velocity reconstruction),
         by calling :func:`~kszx.fft_r2c()` followed by :func:`~kszx.estimate_power_spectrum()`.

       - At some point in the future, I'll define spin-$l$ transforms, with an
         extra factor $(\pm i^l P_l({\hat k} \cdot {\hat r}))$.
    """

    assert isinstance(box, Box)
    assert box.is_real_space_map(arr)   # check shape and dtype of input array

    if threads is None:
        threads = os.cpu_count()
    
    if spin == 0:
        # Currently using scipy.fft instead of pyfftw.
        # (When I tried pyfftw, it was weirdly slow.)
        ret = scipy.fft.rfftn(arr, workers=threads)
        ret *= box.pixel_volume   # see Fourier conventions in docstring
        return ret                # numpy array with shape=box.fourier_space_shape and dtype=complex.

    if spin != 1:
        raise RuntimeError('fft_r2c(): currently we support only spin=0 and spin=1')

    # Spin-1 FFT follows.
    
    ret = np.zeros(box.fourier_space_shape, dtype=complex)
    
    for axis in range(box.ndim):
        t = multiply_rfunc(box, arr, lambda r: 1./r, regulate=True)
        t *= box.get_r_component(axis)
        t = fft_r2c(box, t, spin=0, threads=threads)
        t *= -1j * box.get_k_component(axis, zero_nyquist=True)  # note minus sign here
        ret += t
        del t

    multiply_kfunc(box, ret, lambda k: 1./k, dc=0, in_place=True)
    return ret
    

def fft_c2r(box, arr, spin=0, threads=None):
    r"""Computes the FFT of Fourier-space map 'arr', and returns a real-space map.

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``arr``: numpy array representing a Fourier-space map (dtype=complex).

        - ``spin``: currently only spin=0 and spin=1 are supported.

        - ``threads`` (integer or None): number of parallel threads used.
          If ``threads=None``, then number of threads defaults to ``os.cpu_count()``.

    Returns a numpy array representing a real-space map (dtype=float).

    The real-space and Fourier-space array shapes are given by ``box.real_space_shape``
    and ``box.fourier_space_shape``, and are related as follows:

    $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
    $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

    Notes:

       - Our spin-0 Fourier conventions are (see Box docstring):

          $$f(k) = V_{pix} \sum_x f(x) e^{-ik\cdot x}$$

          $$f(x) = V_{box}^{-1} \sum_k f(k) e^{ik\cdot x}$$

       - We define spin-1 Fourier transforms by inserting an extra factor
         $(\pm i {\hat k} \cdot {\hat r})$:

          $$f(k) = V_{pix} \sum_x f(x) (-i {\hat k} \cdot {\hat r}) e^{-ik\cdot x}$$

          $$f(x) = V_{box}^{-1} \sum_k f(k) (i {\hat k} \cdot {\hat r}) e^{ik\cdot x}$$

         where the line-of-sight direction $\hat r$ is defined in "observer coordinates"
         (see :class:`~kszx.Box` for more info).

         Application: the spin-1 c2r transform can be used to compute the radial velocity
         field from the density field (with a factor $faH/k$).

         Another application: the spin-1 r2c transform can be used to estimate $P_{gv}(k)$
         or $P_{vv}(k)$ from the radial velocity field (or the kSZ velocity reconstruction),
         by calling :func:`~kszx.fft_r2c()` followed by :func:`~kszx.estimate_power_spectrum()`.

       - At some point in the future, I'll define spin-$l$ transforms, with an
         extra factor $(\pm i^l P_l({\hat k} \cdot {\hat r}))$.
    """
    
    assert isinstance(box, Box)
    assert box.is_fourier_space_map(arr)   # check shape and dtype of input array

    if threads is None:
        threads = os.cpu_count()

    if spin == 0:
        # Currently using scipy.fft instead of pyfftw.
        # (When I tried pyfftw, it was weirdly slow.)
        ret = scipy.fft.irfftn(arr, box.npix, workers=threads)
        ret *= (1.0 / box.pixel_volume)    # see Fourier conventions in docstring
        return ret                         # numpy array with shape=box.real_space shape and dtype=complex.

    if spin != 1:
        raise RuntimeError('fft_c2r(): currently we support only spin=0 and spin=1')

    # Spin-1 FFT follows.
    
    ret = np.zeros(box.real_space_shape, dtype=float)
    
    for axis in range(box.ndim):
        t = multiply_kfunc(box, arr, lambda k: 1./k, dc=0.)
        t *= 1j * box.get_k_component(axis, zero_nyquist=True)
        t = fft_c2r(box, t, spin=0)
        t *= box.get_r_component(axis)
        ret += t
        del t

    multiply_rfunc(box, ret, lambda r: 1./r, regulate=True, in_place=True)
    return ret


####################################################################################################


def interpolate_points(box, arr, points, kernel, fft=False, spin=0):
    r"""Interpolates real-space map at a specified set of points.

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``arr`` (numpy array): represents the map to be interppolated, either in real space 
          (if ``fft=False``) or Fourier space (if ``fft=True``).
          
          The real-space and Fourier-space array shapes are given by ``box.real_space_shape``
          and ``box.fourier_space_shape``, and are related as follows:

          $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
          $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

        - ``points`` (numpy array):
          Sequence of points where the map is to be interpolated.
          Array shape should be (n,d), where n is the number of interpolation points, and d
          is the box dimension (usually 3).

        - ``kernel`` (string): either ``'cic'`` or ``'cubic'`` (more options will be defined later).

        - ``fft`` (boolean): if True, then ``arr`` is a Fourier-space map, and ``fft_c2r(arr, spin)``
          will be called before interpolating.
    
        - ``spin`` (integer): passed as 'spin' argument to ``fft_c2r()``. (Only used if ``fft=True``.)

    Return value:

        - 1-d numpy array with length npoints, containing interpolated values.
 
    Note: 

       - The ``points`` array should be specified in "observer coordinates", not "grid coordinates".

         (Reminder: in observer coordinates, the observer is at the origin, coordinates have units
         of distance, and the corners of the box are at ``box.{lpos,rpos}``. 
         See :class:`~kszx.Box` for more info.)
    """

    if not isinstance(box, Box):
        raise RuntimeError("kszx.interpolate_points(): expected 'box' arg to be kszx.Box object, got {box = }")
    if kernel is None:
        raise RuntimeError("kszx.interpolate_points(): 'kernel' arg must be specified")
    if box.ndim != 3:
        raise RuntimeError('kszx.interpolate_points(): currently only ndim==3 is supported')

    arr = utils.asarray(arr, 'kszx.interpolate_points()', 'arr')
    points = utils.asarray(points, 'kszx.interpolate_points()', 'points', dtype=float)
    kernel = kernel.lower()
    
    if (points.ndim != 2) or (points.shape[1] != box.ndim):
        raise RuntimeError(f"kszx.interpolate_points(): expected points.shape=(N,{box.ndim}), got shape {points.shape}")

    is_real_space = box.is_real_space_map(arr)
    is_fourier_space = box.is_fourier_space_map(arr)

    if (not is_real_space) and (not is_fourier_space):
        raise RuntimeError("kszx.interpolate_points(): 'arr' argument has wrong shape/dtype")
    if fft and is_real_space:
        raise RuntimeError("kszx.interpolate_points(): 'arr' argument is real-space map, but fft=True was specified")
    if (not fft) and is_fourier_space:
        raise RuntimeError("kszx.interpolate_points(): 'arr' argument is Fourier-space map, you probably want to specify fft=True")

    if fft:
        arr = fft_c2r(box, arr, spin=spin)
        
    if kernel == 'cic':
        return cpp_kernels.cic_interpolate_3d(arr, points, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)
    elif kernel == 'cubic':
        return cpp_kernels.cubic_interpolate_3d(arr, points, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)
    else:
        raise RuntimeError('kszx.interpolate_points(): {kernel=} is not supported')


def _check_weights(box, points, weights, prefix='', target_sum=None):
    """Helper for grid_points(), used to parse (points,weights) and (rpoints,rweights) args."""

    if (points.ndim != 2) or (points.shape[1] != box.ndim):
        raise RuntimeError(f"kszx.grid_points(): expected {prefix}points.shape=(N,{box.ndim}), got shape {points.shape}")
    
    npoints = points.shape[0]
    
    if weights is None:
        weights = np.ones(npoints)
    elif weights.ndim == 0:
        weights = np.full(npoints, fill_value=float(weights))
    elif weights.shape != (npoints,):
        raise RuntimeError(f"kszx.grid_points(): {prefix}points and {prefix}weights arrays don't have consistent shapes")

    if target_sum is not None:
        rweight_sum = np.sum(weights)
        assert rweight_sum > 0.0
        weights *= (target_sum / rweight_sum)
        
    return weights


def grid_points(box, points, weights=None, rpoints=None, rweights=None, kernel=None, fft=False, spin=0):
    r"""Returns a map representing a sum of delta functions (or a "galaxies - randoms" difference map).

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``points`` (2-d array):
          Sequence of points where the delta functions are located (usually galaxy locations).
          Array shape should be (n,d), where n is the number of interpolation points, and d
          is the box dimension (usually 3).

        - ``weights`` (either scalar, None, or 1-d array).
           - if ``weights`` is an array, then it should have length n, where n is the number of
             interpolation points.
           - if ``weights`` is a scalar, then all delta functions have equal weight.
           - if ``weights`` is None, then all delta functions have weight 1.

        - ``rpoints`` (2-d array or None): 
          Optional sequence of points where delta functions with negative coefficients
          are located (representing a "random" catalog).

        - ``rweights`` (either scalar, None, or 1-d array).
          Weights for the ``rpoints`` array (same semantics as ``weights``).

          NOTE: an additional multiplicative normalization is applied to the ``rweights``
          so that ``sum(rweights) = -sum(weights)``. That is, if ``rweights`` are specified,
          then the (galaxies - randoms) maps returned by this function always has mean zero.

        - ``kernel`` (string): either ``'cic'`` or ``'cubic'`` (more options will be defined later).

        - ``fft`` (boolean): if True, then ``fft_r2c(.., spin)`` will be applied to the
          output map after gridding.
    
        - ``spin`` (integer): passed as 'spin' argument to ``fft_c2r()``. (Only used if ``fft=True``.)
    
    Return value: 

      - A numpy array representing a real-space (``fft=False``) or Fourier-space (``fft=True``) map.

        The real-space and Fourier-space array shapes are given by ``box.real_space_shape``
        and ``box.fourier_space_shape``, and are related as follows:

        $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
        $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

    Note:

       - The ``points`` array should be specified in "observer coordinates", not "grid coordinates".

         (Reminder: in observer coordinates, the observer is at the origin, coordinates have units
         of distance, and the corners of the box are at ``box.{lpos,rpos}``. 
         See :class:`~kszx.Box` for more info.)

       - The normalization of the output map includes a factor (pixel volume)$^{-1}$. 
         (That is, the sum of the output array is $(\sum_j w_j)/V_{\rm pix}$, not $(\sum_j w_j)$.)

         This normalization best represents a weighted sum of delta functions 
         $f(x) = \sum_j w_j \delta^3(x-x_j)$. For example, if we FFT the output array with 
         ``kszx.fft_r2c()``, the result is a weighted sum of plane waves:

         $$f(k) = \sum_j w_j \exp(-i{\bf k}\cdot {\bf x_j})$$
    
         with no factor of box or pixel volume.
    """

    if not isinstance(box, Box):
        raise RuntimeError("kszx.grid_points(): expected 'box' arg to be kszx.Box object, got {box = }")
    if kernel is None:
        raise RuntimeError("kszx.grid_points(): 'kernel' arg must be specified")
    if (rpoints is None) and (rweights is not None):
        raise RuntimeError("kszx.grid_points(): 'rpoints' arg is None, but 'rweights' arg is not None")
    if box.ndim != 3:
        raise RuntimeError('kszx.grid_points(): currently only ndim==3 is supported')

    points = utils.asarray(points, 'kszx.grid_points()', 'points', dtype=float)
    weights = utils.asarray(weights, 'kszx.grid_points()', 'weights', dtype=float, allow_none=True)
    rpoints = utils.asarray(rpoints, 'kszx.grid_points()', 'rpoints', dtype=float, allow_none=True)
    rweights = utils.asarray(rweights, 'kszx.grid_points()', 'rweights', dtype=float, allow_none=True)
    kernel = kernel.lower()
    
    if kernel == 'cic':
        cpp_kernel = cpp_kernels.cic_grid_3d
    elif kernel == 'cubic':
        cpp_kernel = cpp_kernels.cubic_grid_3d
    else:
        raise RuntimeError('kszx.grid_points(): {kernel=} is not supported')        
        
    grid = np.zeros(box.real_space_shape, dtype=float)
    weights = _check_weights(box, points, weights)  # also checks 'points' arg
    cpp_kernel(grid, points, weights, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)

    if rpoints is not None:
        wsum = np.sum(weights)
        rweights = _check_weights(box, rpoints, rweights, prefix='r', target_sum = -wsum)
        cpp_kernel(grid, rpoints, rweights, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)        

    return fft_r2c(box,grid,spin=spin) if fft else grid


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


def _eval_kfunc(box, f, dc=None):
    """Helper for multiply_kfunc(), kbin_average()."""

    assert callable(f)
    assert isinstance(box, Box)

    k = box.get_k(regulate = (dc is not None))
    fk = f(k)

    if fk.shape != box.fourier_space_shape:
        raise RuntimeError('kszx.multiply_kfunc(): function f(k) returned unexpected shape')
    if fk.dtype != float:
        raise RuntimeError('kszx.multiply_kfunc(): function f(k) returned dtype={fk.dtype} (expected float)')

    if dc is not None:
        fk[(0,)*box.ndim] = dc

    return fk


def multiply_rfunc(box, arr, f, dest=None, in_place=False, regulate=False, eps=1.0e-6):
    r"""Multiply real-space map 'arr' by a function f(r), where r is scalar radial coordinate.

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.
    
        - ``arr``: numpy array representing a real-space map. (The array shape should be given by
          ``box.real_space_shape`` and the dtype should be ``float``.)

        - ``f`` (function): The function r -> f(r).

        - ``dest`` (array or None): real-space map where output will be written (if None, then new array will be allocated).

        - ``in_place`` (boolean): Setting this to True is equivalent to ``dest=arr``.

        - ``regulate`` (boolean): This optional argument is intended to regulate cases
          where $\lim_{r\rightarrow 0} f(r) = \infty$. If ``regulate=True``, then we replace r
          by ``max(r, eps*pixsize)`` before calling ``f()``.

        - ``eps`` (float): only used if ``regulate=True``. See description in previous bullet point.

    Return value:

        - A real-space map. (Numpy array with same shape and dtype as ``arr``.)

    Note: 
    
       - The function ``f()`` must be vectorized: its argument 'r' will be a 3-dimensional arary,
         and the return value should be an array with the same shape.

       - r-values passed to ``f()`` will be in "observer" coordinates.

         (Reminder: in observer coordinates, the observer is at the origin, coordinates have units
         of distance, and the corners of the box are at ``box.{lpos,rpos}``. 
         See :class:`~kszx.Box` for more info.)
    """

    assert isinstance(box, Box)
    assert box.is_real_space_map(arr)   # check shape, dtype
    assert callable(f)
    assert eps < 0.5

    r = box.get_r(regulate=regulate, eps=eps)
    fr = f(r)
    
    if not box.is_real_space_map(fr):
        raise RuntimeError('kszx.multiply_rfunc(): function f(r) returned unexpected shape/dtype')

    return _multiply(arr, fr, dest, in_place)
    
    
def multiply_kfunc(box, arr, f, dest=None, in_place=False, dc=None):
    r"""Multiply Fourier-space map 'arr' by a real-valued function f(k), where k=|k| is scalar wavenumber.
    
    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.
    
        - ``arr``: numpy array representing a Fourier-space map. (The array shape should be given by
          ``box.fourier_space_shape`` and the dtype should be ``complex``, see note below.)

        - ``f`` (function): The function k -> f(k).

        - ``dest``: Fourier-space map where output will be written (if None, then new array will be allocated)

        - ``in_place``: Setting this to True is equivalent to ``dest=arr``.

        - ``dc`` (float): This optional argument is intended to regulate cases where
          $\lim_{k\rightarrow 0} f(k) = \infty$. If ``dc`` is specified, then ``f()``
          is not evaluated at k=0, and the value of ``dc`` is used instead of ``f(0)``.


    Return value:

        - A Fourier-space map. (Numpy array with same shape and dtype as ``arr``.)

    Notes: 
    
       - The function ``f()`` must be vectorized: its argument 'k' will be a 3-dimensional arary,
         and the return value should be a real-valued array with the same shape.

       - k-values passed to ``f()`` will be in "physical" units, i.e. the factor ``(2*pi / box.boxsize)``
         is included.

       - The ``arr`` argument and the returned array are Fourier-space maps.
    
         Reminder: real-space and Fourier-space array shapes are given by ``box.real_space_shape``
         and ``box.fourier_space_shape``, and are related as follows:

         $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
         $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$
    """

    assert isinstance(box, Box)
    assert box.is_fourier_space_map(arr)   # check shape, dtype

    fk = _eval_kfunc(box, f, dc=dc)
    return _multiply(arr, fk, dest, in_place)


def multiply_r_component(box, arr, axis, dest=None, in_place=True):
    r"""Multiply real-space map 'arr' by $r_j$ (the j-th Cartesian coordinate).

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.
    
        - ``arr``: numpy array representing a real-space map. (The array shape should be given by
          ``box.real_space_shape`` and the dtype should be ``float``.)

        - ``axis`` (integer): axis j (satisfying ``0 <= j < box.ndim``) along which $r_j$ is computed.

        - ``dest`` (array or None): real-space map where output will be written (if None, then new array will be allocated).

        - ``in_place`` (boolean): Setting this to True is equivalent to ``dest=arr``.

    Return value:

        - A real-space map. (Numpy array with same shape and dtype as ``arr``.)

    Note: 

       - Values of $r_j$ will be signed, and in "observer" coordinates.

         (Reminder: in observer coordinates, the observer is at the origin, coordinates have units
         of distance, and the corners of the box are at ``box.{lpos,rpos}``. 
         See :class:`~kszx.Box` for more info.)
    """
    
    assert isinstance(box, Box)
    assert box.is_real_space_map(arr)

    ri = box.get_r_component(axis)
    return _multiply(arr, ri, dest, in_place)

    
def apply_partial_derivative(box, arr, axis, dest=None, in_place=True):
    r"""Multiply Fourier-space map 'arr' by $(i k_j)$. (This is the partial derivative $\partial_j$ in Fourier space.)

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.
        
        - ``arr``: numpy array representing a Fourier-space map. (The array shape should be given by
          ``box.fourier_space_shape`` and the dtype should be ``complex``, see note below.)

        - ``axis`` (integer): axis j (satisfying ``0 <= j < box.ndim``) along which $(i k_j)$ is computed.

        - ``dest`` (array or None): Fourier-space map where output will be written (if None, then new array will be allocated).

        - ``in_place`` (boolean): Setting this to True is equivalent to ``dest=arr``.

    Return value:

        - A Fourier-space map. (Numpy array with same shape and dtype as ``arr``.)

    Notes: 

       - k-values passed to ``f()`` will be in "physical" units, i.e. the factor ``(2*pi / box.boxsize)``
         is included.
    
       - Values of $k_j$ will be signed, and include the factor (2pi / boxsize).

       - The value of k_j will be taken to be zero at the Nyquist frequency.
         (I think this is the only sensible choice, since the sign is ambiguous.)

       - The ``arr`` argument and the returned array are Fourier-space maps.
    
         Reminder: real-space and Fourier-space array shapes are given by ``box.real_space_shape``
         and ``box.fourier_space_shape``, and are related as follows:

         $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
         $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$
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
            raise RuntimeError('kszx.simulate_gaussian_field(): function pk() returned unexpected shape')
        if pk.dtype != float:
            raise RuntimeError('kszx.simulate_gaussian_field(): function pk() returned dtype={pk.dtype} (expected float)')
        if np.min(pk) < 0:
            raise RuntimeError('kszx.simulate_gaussian_field(): function pk() returned negative values')

        del k
        pk **= 0.5
        return pk   # returns sqrt(P(k))

    pk = _to_float(pk, 'kszx.simulate_gaussian_field(): expected pk argument to be either callable, or a real scalar')

    if pk < 0:
        raise RuntimeError('kszx.simulate_gaussian_field(): expected scalar pk argument to be non-negative')
    
    return np.sqrt(pk)

    
def simulate_white_noise(box, *, fourier):
    r"""Simulate white noise, in either real space or Fourier space, normalized to $P(k)=1$.

    Intended as a helper for ``simulate_gaussian_field()``, but may be useful on its own.

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``fourier`` (boolean): determines whether output is real-space or Fourier-space.
    
    Return value: 

      - A numpy array representing a real-space (``fourier=False``) or Fourier-space (``fourier=True``) map.

        The real-space and Fourier-space array shapes are given by ``box.real_space_shape``
        and ``box.fourier_space_shape``, and are related as follows:

        $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
        $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

    Note: our normalization conventions for the simulated field are (in Fourier and real space):
    
    $$\langle f(k) f(k')^* \rangle = V_{\rm box} \delta_{kk'}$$
    $$\langle f(x) f(x') \rangle = V_{\rm pix}^{-1} \delta_{xx'}$$
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
    r"""Simulates a Gaussian field (in Fourier space) with specified power spectrum P(k).

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``pk`` (function or scalar): The power spectrum, represented as a function $k \rightarrow P(k)$.
          If the power spectrum is constant in $k$, then a scalar can be used instead of a function.

        - ``pk0`` (scalar or None): This optional argument is intended to regulate cases
          where $\lim_{k\rightarrow 0} P(k) = \infty$. If ``pk0`` is specified, then ``pk()`` is
          not evaluated at k=0, and the value of ``pk0`` is used instead of ``Pk(0)``.
    
    Return value: 

         - A numpy array representing a Fourier-space map. (Array shape is given by
           ``box.fourier_space_shape``, and dtype is complex, see note below.)

    Notes:

       - The normalization of the simulated field is:

         $$\langle f(k) f(k')^* \rangle = V_{\rm box} P(k) \delta_{kk'}$$
    
       - The function ``pk()`` must be vectorized: its argument 'k' will be a 3-dimensional arary,
         and the return value should be a real-valued array with the same shape.
    
       - k-values passed to ``pk()`` will be in "physical" units, i.e. the factor ``(2*pi / box.boxsize)``
         is included.

       - The returned array is a Fourier-space map.
    
         Reminder: real-space and Fourier-space array shapes are given by ``box.real_space_shape``
         and ``box.fourier_space_shape``, and are related as follows:

         $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
         $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$
    """
    
    r"""Simulates a Gaussian field with specified power spectrum P(k).

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

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
        pk0 = _to_float(pk0, 'kszx.simulate_gaussian_field(): expected pk0 argument to be a real scalar')
        if pk0 < 0:
            raise RuntimeError('kszx.simulate_gaussian_field(): expected pk0 argument to be non-negative')
        ret[(0,)*box.ndim] = np.sqrt(pk0) * dc
    
    return ret


####################################################################################################


def _check_kbin_delim(box, kbin_delim, use_dc):
    """Helper for estimate_power_spectrum() and kbin_average(). Returns new kbin_delim."""

    kbin_delim = np.asarray(kbin_delim, dtype=float)
    
    assert isinstance(box, Box)
    assert kbin_delim.ndim == 1
    assert len(kbin_delim) >= 2    
    assert kbin_delim[0] >= 0.
    assert utils.is_sorted(kbin_delim)
        
    if (not use_dc) and (kbin_delim[0] == 0):
        kbin_delim = np.copy(kbin_delim)
        kbin_delim[0] = min(np.min(box.kfund), kbin_delim[1]) / 2.

    return kbin_delim
    
    
def _parse_map_or_maps(box, map_or_maps):
    """Helper for estimate_power_spectrum(). Returns (map_list, multi_map_flag)."""

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
    r"""Computes power spectrum $P(k)$ for one or more maps (including cross-spectra). The window function is not deconvolved.

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``map_or_maps`` (array or list of arrays): single or multiple Fourier-space maps.

           - If ``map_or_maps`` is an array, then it represents a single Fourier-space map.
             (The array shape should be given by ``box.fourier_space_shape`` and the dtype should be 
             ``complex``, see note below.)
    
           - If ``map_or_maps`` is a list of arrays, then it represents multiple Fourier-space maps.
             (Each map in the list should have shape ``box.fourier_space_shape`` and dtype ``complex``,
             see note below.)

        - ``kbin_delim`` (1-d array): 1-d array of length (nkbins+1) defining bin endpoints.
          The i-th bin covers k-range ``kbin_delim[i] <= k < kbin_delim[i+1]``.

        - ``use_dc`` (boolean): if False (the default), then the k=0 mode will not be used,
          even if the lowest bin includes k=0.

        - ``allow_empty_bins`` (boolean): if False (the default), then an execption is thrown 
          if a k-bin is empty.

        - ``return_counts`` (boolean): See below.

    Return value: 

       - An array ``pk`` is returned, with two cases as follows:

           - If the ``map_or_maps`` argument is a single Fourier-space map (see above),
             then ``pk`` is a 1-d array with length ``nkbins``, containing binned power
             spectrum estimates.

           - If the ``map_or_maps`` argument is a list of Fourier-space maps (see above),
             then ``pk`` is a 3-d array with shape ``(nmaps, nmaps, nkbins)``, containing
             all auto- and cross-power spectrum estimates.

       - If ``return_counts=False`` (the default), then the return value is simply the ``pk`` array.

         If ``return_counts=True``, then the return value is a pair ``(pk, bin_counts)``, where
         ``bin_counts`` is a 1-d array with length ``nkbins``, containing the number of Fourier
         modes in each k-bin.

    Notes:

       - Normalization: to estimate the power spectrum, we square each Fourier mode and divide by 
         the box volume. This is consistent with our normalization for Fourier-space maps (see e.g.
         the :func:`~kszx.simulate_gaussian_field` docstring), which is:

         $$\langle f(k) f(k')^* \rangle = V_{\rm box} P(k) \delta_{kk'}$$
    
       - The normalization of the estimated power spectrum P(k) assumes that the maps fill the
         entire box volume. If this is not the case (i.e. there is a survey window) then you'll
         need to renormalize P(k) or deconvolve the window function.

       - The input arrays are Fourier-space maps.
    
         Reminder: real-space and Fourier-space array shapes are given by ``box.real_space_shape``
         and ``box.fourier_space_shape``, and are related as follows:

         $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
         $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$
    """

    kbin_delim = _check_kbin_delim(box, kbin_delim, use_dc)
    map_list, multi_map_flag = _parse_map_or_maps(box, map_or_maps)
    
    if len(map_list) == 0:
        raise RuntimeError("kszx.estimate_power_spectrum(): expected 'map_or_maps' arg to be either"
                           + "a Fourier-space map, or an iterable returning Fourier-space maps")

    
    if len(map_list) > 4:
        # Note: If the C++ code is modified to allow larger numbers of maps,,
        # make sure to update the unit test too (test_estimate_power_spectrum()).
        raise RuntimeError("kszx.estimate_power_spectrum(): we currently only support nmaps <= 4."
                           + " This is a temporary problem that I'll fix later. It needs minor changes"
                           + " to the C++ code.")
    
    pk, bin_counts = cpp_kernels.estimate_power_spectrum(map_list, kbin_delim, box.npix, box.kfund, box.box_volume)

    if (not allow_empty_bins) and (np.min(bin_counts) == 0):
        raise RuntimeError('kszx.estimate_power_spectrum(): some k-bins were empty')
    
    if not multi_map_flag:
        pk = pk[0,0,:]   # shape (1,1,nkbins) -> shape (nkbins,)

    return (pk, bin_counts) if return_counts else pk


def kbin_average(box, f, kbin_delim, *, use_dc=False, allow_empty_bins=False, return_counts=False):
    """Averages a real-valued function f(k) in k-bins.

    Args:

        box (kszx.Box): defines pixel size, bounding box size, and location relative to observer.

        f: function (or callable object) representing the function k -> f(k).

        kbin_delim: 1-d array of length (nkbins+1) defining bin endpoints.
          The i-th bin covers k-range kbin_delim[i] <= k < kbin_delim[i+1].

        use_dc: if False (the default), then skip k=0, and don't evaluate f(k) at k=0.

        allow_empty_bins: if False (the default), then an execption is thrown if a k-bin is empty.

        return_counts (boolean): See below.

    Returns: 

       - if return_counts=False (the default), then return value is a length-nkbins array 'fk_mean'.
       - if return_counts=True, then return value is a pair (fk_mean, bin_counts).

    Notes: 

       - This function is intended to be used in situations where we want to compare the
         output of estimate_power_spectrum() to a "theory" power spectrum, such as CAMB Plin().
         To remove binning artifacts, you may want to bin-average the theory power spectrum
         over the same k-bins used in estimate_power_spectrum().
    
       - The function f() must be vectorized: its argument 'k' will be a 3-dimensional arary,
         and the return value should be a real-valued array with the same shape.

       - k-values passed to f() will include the factor (2pi / boxsize).

       - The output of kbin_average() does not include any normalization factors (such as the
         box or pixel size). It is just a straightforward average of f() values over each k-bin.
    """

    kbin_delim = _check_kbin_delim(box, kbin_delim, use_dc)
    fk = _eval_kfunc(box, f, dc = (None if use_dc else 0.))

    fk_mean, bin_counts = cpp_kernels.kbin_average(fk, kbin_delim, box.npix, box.kfund)

    if (not allow_empty_bins) and (np.min(bin_counts) == 0):
        raise RuntimeError('kszx.kbin_average(): some k-bins were empty')

    return (fk_mean, bin_counts) if return_counts else fk_mean


####################################################################################################


def fkp_from_ivar_2d(ivar, cl0, normalize=True, return_wvar=False):
    """Makes a 2-d pixell FKP map (intended for CMB) from a pixell inverse variance map.

    The 'ivar' argument is an inverse noise varaiance map (i.e. pixell.enmap) in (uK)^{-2},
    for example the return value of act.read_ivar().

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


def ivar_combine(ivar1, ivar2):
    """Given two 2-d pixell ivar maps, return the 'combined' ivar map (1/ivar1 + 1/ivar2)^{-1}."""
    
    assert isinstance(ivar1, pixell.enmap.ndmap)
    assert isinstance(ivar2, pixell.enmap.ndmap)

    den = ivar1 + ivar2    
    den = np.where(den > 0, den, 1.0)
    num = ivar1 * ivar2
    
    num /= den
    return num


def estimate_cl(alm_or_alms, lbin_delim):
    """Similar interface to estimate_power_spectrum(), but for all-sky alms.

    It's okay if lbin_delim is a float array (it will be converted to int).

    If 'alm_or_alms' is a 1-d array (single alm), returns a 1-d array of length (nlbins,).
    If 'alm_or_alms' is a 2-d array (multiple alms), returns a 3-d array of shape (nalm,nalm,nlbins).
    """

    alm_or_alms = utils.asarray(alm_or_alms, 'kszx.estimate_cl()', 'alm_or_alms')
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
    
    # Check 'alm_or_alms' arg and convert to 2-d.
    if (alm_or_alms.dtype != complex) and (alm_or_alms.dtype != np.complex64):
        raise RuntimeError("kszx.estimate_cl(): 'alm_or_alms' array did not have complex dtype")
    if alm_or_alms.ndim == 1:
        alm_or_alms = np.reshape(alm_or_alms, (1,-1))
        multi_flag = False
    elif alm_or_alms.ndim != 2:
        raise RuntimeError("kszx.estimate_cl(): 'alm_or_alms' array did not have expected shape")

    alm_or_alms = np.asarray(alm_or_alms, dtype=complex)   # convert complex64 -> complex128
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
