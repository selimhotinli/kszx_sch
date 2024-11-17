"""The ``kszx.utils`` module contains miscelleanous utilities, that didn't really fit in elsewhere."""

import numpy as np
import scipy.special
import scipy.integrate
import scipy.interpolate


####################################################################################################


def ra_dec_to_xyz(ra_deg, dec_deg, r=None):
    """Converts spherical polar coordinates (ra_deg, dec_deg) to cartesian coords (x,y,z).
    
    Function arguments:
    
       - ``ra_deg`` (array): RA in degrees (not radians!)
       - ``dec_deg`` (array): DEC in degrees (not radians!)
       - ``r`` (optional array): radial coordinates (units of length)

    Return value:

       - ``xyz`` (array): new array with a length-3 axis appended.
         (E.g. if ra_deg and dec_deg have shape (m,n), then xyz has shape (m,n,3)).

    The coordinate systems are related by::

        x = r * cos(dec) * cos(ra)
        y = r * cos(dec) * sin(ra)
        z = r * sin(dec)
    """
    
    ra_deg = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)
    assert ra_deg.shape == dec_deg.shape

    # FIXME memory-inefficent (uses too many temp arrays)
    
    ra = ra_deg * (np.pi/180.)
    dec = dec_deg * (np.pi/180.)
    cos_dec = np.cos(dec)
    
    ret = np.empty(ra_deg.shape + (3,), dtype=float)
    ret[...,0] = cos_dec * np.cos(ra)
    ret[...,1] = cos_dec * np.sin(ra)
    ret[...,2] = np.sin(dec)

    if r is not None:
        r = np.asarray(r)
        assert r.shape == ra_deg.shape
        ret *= np.reshape(r, r.shape + (1,))
    
    return ret


def xyz_to_ra_dec(xyz, return_r=False):
    """Converts cartesian coords (x,y,z) to spherical polar coordinates (ra_deg, dec_deg).

    Function arguments:
    
       - ``xyz`` (array): array whose last axis has length 3.
       - ``return_r`` (boolean): indicates whether radial coords are returned, see below.

    Return value:

       - ``ra_deg`` (array): RA in degrees (not radians!)
       - ``dec_deg`` (array): DEC in degrees (not radians!)
       - ``r`` (array): radial coordinates (in same units as ``xyz``)
       - If ``return_r`` is True, then the return value is a triple ``(ra_deg, dec_deg, r)``.
         If ``return_r`` is False, then the return value is a pair ``(ra_deg, dec_deg)``.

    The returned arrays have one lower dimension than ``xyz``. (E.g. if ``xyz`` has shape (m,n,3)
    then ``ra_deg``, ``dec_deg``, and ``r`` all have shape (m,n).)

    The coordinate systems are related by::

        x = r * cos(dec) * cos(ra)
        y = r * cos(dec) * sin(ra)
        z = r * sin(dec)
    """

    xyz = np.asarray(xyz)
    assert xyz.shape[-1] == 3

    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    ra_deg = np.arctan2(y, x)
    ra_deg *= (180./np.pi)
    
    dec_deg = np.arctan2(z, np.sqrt(x*x + y*y))
    dec_deg *= (180./np.pi)

    if return_r:
        r = np.sqrt(x*x + y*y + z*z)
        return ra_deg, dec_deg, r
    else:
        return ra_deg, dec_deg
    

####################################################################################################


def W_tophat(x):
    r"""Returns Fourier transform of a 3-d tophat W(x), where x=kR. Vectorized.
    
    W(x) is given by any of the equivalent forms:

    .. math::

       W(x) &= 3 (\sin(x) - x \cos(x)) / x^3 \\
            &= 3 j_1(x) / x \\
            &= j_0(x) + j_2(x)
    """

    # Timing showed that this implementation was fastest.
    mask = (x != 0)   # to regulate division by zero
    ret = 3 * scipy.special.spherical_jn(1,x) / np.where(mask,x,1.0)
    return np.where(mask, ret, 1.0)


####################################################################################################


def asarray(x, caller, arg, dtype=None, allow_none=False):
    if allow_none and (x is None):
        return x
    
    try:
        assert x is not None
        return np.asarray(x, dtype)
    except:
        pass

    s = f' with dtype {dtype}' if (dtype is not None) else ''
    raise RuntimeError(f"{caller}: couldn't convert {arg} to array{s} (value={x})")
    

def scattered_add(lhs, ix, rhs, normalize_sum = None):
    """Returns sum of RHS values (before applying 'normalize_sum').
    FIXME: Is there a numpy function that does this? If not, write a C++ kernel.
    """
    
    assert lhs.ndim == 1
    assert ix.ndim == 1

    n = len(ix)
    rhs = np.asarray(rhs) if (rhs is not None) else None

    # Split lhs into (scalar, vector) components (the vector can be None).
    if rhs is None:
        rs, rv = 1.0, None
    elif rhs.ndim == 0:
        rs, rv = float(rhs), None
    elif rhs.shape == (n,):
        rs, rv = 1.0, rhs
    else:
        raise RuntimeError("kszx.utils.scattered_add(): 'rhs' and 'ix' have inconsistent shapes")

    rsum = (rs*np.sum(rv)) if (rv is not None) else (rs*n)
    
    if normalize_sum:
        assert rsum > 0.0
        rs *= (normalize_sum / rsum)
        
    # FIXME slow non-vectorized loops, write C++ helper function at some point.
    # (I don't think there is a numpy function that helps.)

    if rv is None:
        for i in range(n):
            lhs[ix[i]] += rs
    elif rs == 1.0:
        for i in range(n):
            lhs[ix[i]] += rv[i]
    else:
        for i in range(n):
            lhs[ix[i]] += rs * rv[i]

    return rsum
            

####################################################################################################


def quad(f, xmin, xmax, epsabs=0.0, epsrel=1.0e-4, points=None):
    """Wrapper for scipy.integrate.quad()."""
    
    assert xmin < xmax

    if points is not None:
        # Remove points outside range (not sure if scipy does this automatically)
        points = [ x for x in points if (xmin < x < xmax) ]                                       
    
    return scipy.integrate.quad(f, xmin, xmax, epsabs=epsabs, epsrel=epsrel)[0]


def dblquad(f, xmin, xmax, ymin, ymax, *, epsabs=0.0, epsrel=1.0e-4):
    """
    Wrapper for scipy.integrate.dblquad().
      - The 'f' argument is f(x,y) [not f(y,x) as in scipy.integrate.dblquad]
      - The 'ymin' and 'ymax' arguments can either be floats or functions of x
    """

    ff = lambda x,y: f(y,x)  # swap
    return scipy.integrate.dblquad(f, xmin, xmax, ymin, ymax, epsabs=epsabs, epsrel=epsrel)[0]


def spline1d(xvec, yvec):
    """Wrapper for scipy.interpolate.InterpolatedUnivariateSpline()."""
    
    assert xvec.shape == yvec.shape
    assert xvec.ndim == yvec.ndim == 1
    return scipy.interpolate.InterpolatedUnivariateSpline(xvec, yvec)


def spline2d(xvec, yvec, zmat):
    """
    Wrapper for scipy.interpolate.RectBivariateSpline().
    
    Note that RectBivariateSpline.__call__(x,y) takes a 'grid' argument:

       grid=False  means "broadcast x,y to get a set of points"
       grid=True   means "take the Cartesian product of x,y to get a grid"

    Warning: grid=True is the default! (I usually want grid=False).
    """

    assert xvec.ndim == yvec.ndim == 1
    assert zmat.shape == (len(xvec), len(yvec))
    return scipy.interpolate.RectBivariateSpline(xvec, yvec, zmat)



def logspace(xmin, xmax, n=None, dlog=None):
    """
    Returns a 1-d array of values, uniformly log-spaced over the range (xmin, xmax).
    
    The spacing can be controlled by setting either the 'n' argument (number of points)
    or 'dlog' (largest allowed spacing in log(x)).
    """

    assert 0 < xmin < xmax
    assert (n is None) or (n >= 2)

    if (n is None) and (dlog is not None):
        n = int((np.log(xmax) - np.log(xmin)) / dlog) + 2
    elif (n is None) or (dlog is not None):
        raise RuntimeError("logspace: exactly one of 'n', 'dlog' should be None")

    ret = np.exp(np.linspace(np.log(xmin), np.log(xmax), n))
    ret[0] = xmin   # get rid of roundoff error
    ret[-1] = xmax  # get rid of roundoff error
    
    return ret


def log_uniform(xmin, xmax, size=None):
    assert 0 < xmin < xmax
    return np.exp(np.random.uniform(np.log(xmin), np.log(xmax), size=size))


def covscale(m):
    m = np.array(m)
    assert m.ndim == 2
    assert m.shape[0] == m.shape[1]
    assert np.all(m.diagonal() > 0.0)

    n = m.shape[0]
    ret = np.zeros((n,n))

    for i in xrange(n):
        for j in xrange(n):
            ret[i,j] = m[i,j] / (m[i,i]*m[j,j])**0.5

    return ret


def is_perfect_square(n):
    if n < 0:
        return False
    m = int(np.round(np.sqrt(n)))
    return n == m**2


def is_sorted(x):
    assert x.ndim == 1
    assert len(x) >= 2
    return np.all(x[:-1] < x[1:])
    

def trapezoid_weights(xmin, xmax, n):
    """Returns xvec, wvec."""

    assert xmin < xmax
    assert n >= 2
    
    xvec = np.linspace(xmin, xmax, n)
    wvec = np.full(n, (xmax-xmin)/(n-1))
    wvec[0] *= 0.5
    wvec[-1] *= 0.5

    return xvec, wvec


def array_slice(arr, axis, *args):
    assert 0 <= axis < arr.ndim
    t1 = (slice(None),) * axis
    t2 = (slice(*args),)
    return a[t1+t2]
        
        
def range_checked_index_operator(source_array, index_array, default_value=None, error_message=None):
    """Returns source_array[index_array], but substitutes 'default_value' if index is out of range."""
    
    assert source_array.ndim == 1
    assert len(source_array) > 0

    n = len(source_array)
    valid = np.logical_and(index_array >= 0, index_array < n)

    if default_value is not None:
        index_array = np.where(valid, index_array, 0)
        return np.where(valid, source_array[index_array], default_value)
    
    if np.all(valid):
        return source_array[index_array]

    if error_message is None:
        error_message = 'index out of range, and no default_value specfied'

    raise RuntimeError(error_message)


def contains_duplicates(l):
    return len(l) != len(set(l))


def boxcar_sum(a, m, normalize=False):
    """
    Given a 1-d array 'a' of length n, return a length-n array containing boxcar-summed values of a.
    Each boxcar consists of (2*m+1) array elements, or fewer near the edges of the array.

    E.g. if m=2, then

      ret[0] = a[0] + a[1] + a[2]
      ret[1] = a[0] + a[1] + a[2] + a[3]
      ret[2] = a[0] + a[1] + a[2] + a[3] + a[4]
      ret[3] = a[1] + a[2] + a[3] + a[4] + a[5]
        ...
      ret[n-4] = a[n-6] + a[n-5] + a[n-4] + a[n-3] + a[n-2]
      ret[n-3] = a[n-5] + a[n-4] + a[n-3] + a[n-2] + a[n-1]
      ret[n-2] = a[n-4] + a[n-3] + a[n-2] + a[n-1]
      ret[n-1] = a[n-3] + a[n-2] + a[n-1]
    
    If 'normalize' is True, then "boxcar means" are computed instead of "boxcar sums".
    """
    
    a = np.asarray(a)
    assert a.ndim == 1
    assert m >= 0

    if m == 0:
        return np.copy(a)
    
    n = len(a)
    assert n > 0
        
    if n <= m+1:
        t = np.mean(a) if normalize else np.sum(a)
        return np.full((n,), t, dtype=a.dtype)

    c = np.cumsum(a)
    ret = np.zeros(n, dtype=a.dtype)
    ret[:-m] = c[m:]
    ret[-m:] = c[-1]
    ret[(m+1):] -= c[:(-m-1)]
    
    if normalize:
        den = np.full((n,), 2*m+1)
        den[:m] -= np.arange(m,0,-1)
        den[-m:] -= np.arange(1,m+1)
        ret = ret / den

    return ret
