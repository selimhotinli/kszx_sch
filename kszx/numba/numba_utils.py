from numba import njit, prange, types, set_num_threads, vectorize
import numpy as np
from numba.experimental import jitclass
import os
from math import erfc

set_num_threads(os.cpu_count())

spec = [
    ('x', types.float64[:]),
    ('y', types.float64[:]),
    ('a', types.float64[:]),
    ('b', types.float64[:]),
    ('c', types.float64[:]),
    ('d', types.float64[:]),
    ('n', types.int64),
    ('loglog', types.boolean),
    ('eps', types.float64),
]

@jitclass(spec)
class CubicSpline1D:
    def __init__(self, x, y, loglog=False, eps=1e-10):
        """
        Natural cubic spline interpolator
        """
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        self.n = len(x) - 1
        self.loglog = loglog
        self.eps = float(eps)
        if self.loglog:
            x_safe = np.maximum(x, 0.0) + self.eps
            y_safe = np.maximum(y, 0.0) + self.eps
            self.x = np.log10(x_safe)
            self.y = np.log10(y_safe)
        else:
            self.x = x
            self.y = y
            
        self.a = np.zeros(self.n, dtype=np.float64)
        self.b = np.zeros(self.n, dtype=np.float64)
        self.c = np.zeros(self.n + 1, dtype=np.float64)
        self.d = np.zeros(self.n, dtype=np.float64)
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        n = self.n
        x = self.x
        y = self.y
        h = np.zeros(n, dtype=np.float64)
        alpha = np.zeros(n, dtype=np.float64)
        l = np.zeros(n + 1, dtype=np.float64)
        mu = np.zeros(n + 1, dtype=np.float64)
        z = np.zeros(n + 1, dtype=np.float64)
    
        # Step 1: Calculate h and alpha
        for i in range(n):
            h[i] = x[i+1] - x[i]
        for i in range(1, n):
            alpha[i] = (3.0/h[i]) * (y[i+1] - y[i]) - (3.0/h[i-1]) * (y[i] - y[i-1])
    
        # Step 2: Set up the tridiagonal system
        l[0] = 1.0
        mu[0] = 0.0
        z[0] = 0.0
        for i in range(1, n):
            l[i] = 2.0*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
    
        l[n] = 1.0
        z[n] = 0.0
        self.c[n] = 0.0
    
        # Step 3: Back substitution
        for j in range(n-1, -1, -1):
            self.c[j] = z[j] - mu[j]*self.c[j+1]
            self.b[j] = (y[j+1] - y[j])/h[j] - h[j]*(self.c[j+1] + 2.0*self.c[j])/3.0
            self.d[j] = (self.c[j+1] - self.c[j])/(3.0*h[j])
            self.a[j] = y[j]
    
    def evaluate(self, xi):
        return evaluate_point(xi, self)

@njit
def evaluate_point(xi, spline):
    x = spline.x
    n = spline.n
    a = spline.a
    b = spline.b
    c = spline.c
    d = spline.d
    loglog = spline.loglog
    eps = spline.eps

    if loglog:
        xi_safe = np.maximum(xi, 0.0) + eps
        xi_eval = np.log10(xi_safe)
    else:
        xi_eval = xi

    idx = np.searchsorted(x, xi_eval) - 1

    # Handle extrapolation (we will extrapolate linearly)
    if idx < 0:
        idx = 0
        dx = xi_eval - x[0]
        spline_value = a[idx] + b[idx]*dx
    elif idx >= n:
        idx = n - 1
        dx = xi_eval - x[n]
        spline_value = a[idx] + b[idx]*dx
    else:
        dx = xi_eval - x[idx]
        spline_value = a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
        
    if loglog:
        result = np.power(10.0, spline_value)
    else:
        result = spline_value
    return result

@njit(parallel=True)
def evaluate_spline_parallel(spline, xi_array):
    n_points = xi_array.size
    result = np.empty_like(xi_array)
    for i in prange(n_points):
        xi = xi_array.flat[i]
        result.flat[i] = evaluate_point(xi, spline)
    return result

@njit
def evaluate_spline_sequential(spline, xi_array):
    n_points = xi_array.size
    result = np.empty_like(xi_array)
    for i in range(n_points):
        xi = xi_array.flat[i]
        result.flat[i] = evaluate_point(xi, spline)
    return result


class spline1d:
    def __init__(self, x_arr, y_arr, loglog=False):
        self.spline = CubicSpline1D(x_arr, y_arr, loglog)
    def __call__(self, xi_arr):
        if len(xi_arr.shape) == 0:
            return evaluate_point(np.atleast_1d(xi_arr), self.spline)    
        elif xi_arr.size <= int(1e4):
            return evaluate_spline_sequential(self.spline, xi_arr)
        else:
            return evaluate_spline_parallel(self.spline, xi_arr)
    
#@njit(parallel=True)
#def _parallel_check_bounds_incl(x, lb, rb):
#    for i in prange(x.size):
#        if (x.flat[i] < lb) or (x.flat[i] > rb):
#            return False
#        return True
    
@njit
def check_bounds_incl(x, lb, rb):
    for i in prange(x.size):
        if (x.flat[i] < lb) or (x.flat[i] > rb):
            return False
        return True
    
#def check_bounds_incl(x, lb, rb):
#    if x.size > int(1e4):
#        return _parallel_check_bounds_incl(x, lb, rb)
#    return _check_bounds_incl(x, lb, rb)

@vectorize([types.float32(types.float32), 
            types.float64(types.float64)], nopython=True, target='parallel')
def _exp(arr):
    return np.exp(arr)


@vectorize([types.float32(types.float32), 
            types.float64(types.float64)], nopython=True, target='parallel')
def _sqrt(arr):
    return np.sqrt(arr)

@njit(parallel=True)
def _random_normal(out, scale):
    for i in prange(out.size):
        out.flat[i] = np.random.normal(scale=scale, loc=0.)
    return out    

def random_normal(size, scale):
    scale = float(scale)
    out = np.empty(size, dtype=np.float64)
    return _random_normal(out, scale)
    
    
@njit(parallel=True)    
def multiply_inplace(arr_a, arr_b):
    for i in prange(arr_a.size):
        arr_a.flat[i] *= arr_b.flat[i]
        
        
@njit
def get_k_value(i, j, k_index, dim, L):
    """
    Given grid indices (i, j, k_index), grid resolution 'dim', and box size 'L',
    returns the corresponding k value in the Fourier grid.

    Parameters:
    - i, j: Grid indices along the x and y axes (0 <= i, j < dim)
    - k_index: Grid index along the z-axis (0 <= k_index <= dim // 2)
               (since we're using a reduced FFT along this axis)
    - dim: Grid resolution along each axis
    - L: Physical size of the box along each axis

    Returns:
    - k_value: The corresponding k value
    """
    n = dim
    if i < n // 2:
        freq_i = 2 * np.pi * i / L
    else:
        freq_i = 2 * np.pi * (i - n) / L

    if j < n // 2:
        freq_j = 2 * np.pi * j / L
    else:
        freq_j = 2 * np.pi * (j - n) / L

    freq_k = 2 * np.pi * k_index / L

    k_squared = freq_i**2 + freq_j**2 + freq_k**2

    return np.sqrt(k_squared)

@njit(parallel=True)
def get_k_3D_box(dim, L):
    """ Generates a grid of fourier k in a 3d box with 'dim' pixels per side 'L'
    """
    out = np.empty((dim,dim,dim//2+1),dtype=np.float64)
    for i in prange(dim):
        for j in range(dim):
            for k in range(dim//2+1):
                out[i,j,k] = get_k_value(i,j,k, dim, L).item()
    return out



@njit
def _cic_ker(x):
    return 1. - (2./3.)*x**2

@njit
def _cubic_ker(x):
    return 1. - (22./45.)*(x**4) - (124./945.)*(x**6)

@njit(parallel=True)
def _apply_kernel_compensation(arr, ker_f, exponent):
    
    imax, jmax, kmax = arr.shape
    rdim = imax
    for i in prange(imax):
        for j in range(jmax):
            for k in range(kmax):
                
                si = np.sin(np.pi * i / rdim)
                sj = np.sin(np.pi * j / rdim)
                sk = np.sin(np.pi * k / rdim)
                
                compensation = ker_f(si)*ker_f(sj)*ker_f(sk)
                
                arr[i,j,k] *= np.power(compensation, exponent) 
    
    

@vectorize([types.float32(types.float32),
            types.float64(types.float64)],
           nopython=True,
            target='parallel')
def _P(x):
    return 0.5*erfc(x/2**0.5)      