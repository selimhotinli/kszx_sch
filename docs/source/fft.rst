.. _fft wrappers:

:mod:`FFT wrappers`
===================

.. autofunction:: kszx.fft_r2c

.. autofunction:: kszx.fft_c2r

.. _fft_conventions:

Conventions and "spin"
----------------------

In this section, we document our conventions/normalizations for FFTs, and the meaning of the
``spin`` argument to :func:`~kszx.fft_c2r()` and :func:`~kszx.fft_r2c()`.

Real-space and Fourier-space maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 - A real-space map is represented as a pair ``(box, numpy_array)``, where ``box`` is an instance
   of class :class:`~kszx.Box`. The numpy array has ``float`` dtype and shape:

   $$(\mbox{real-space shape}) = {\tt \mbox{box.real_space_shape}} = (n_0, n_1, \cdots, n_{d-1})$$

 - A Fourier-space map is represented as a pair ``(box, numpy_array)``, where ``box`` is an instance
   of class :class:`~kszx.Box`. The numpy array has ``complex`` dtype and shape:

   $$(\mbox{Fourier-space shape}) = {\tt \mbox{box.fourier_space_shape}} = (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

 - Note that we define a Box class, but not a Map class (instead, we represent maps by ``(box,arr)`` pairs).
   For now, ``arr`` must be an ordinary numpy array, but in the future, we might support more fun possibilities
   (e.g. mpi/cupy/jax arrays).

Fourier conventions and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 - In :func:`~kszx.fft_c2r()` and :func:`~kszx.fft_r2c()`, we use the following Fourier conventions:

   $$\begin{align}
   f(k) &= V_{pix} \sum_x f(x) e^{-ik\cdot x} \\
   f(x) &= V_{box}^{-1} \sum_k f(k) e^{ik\cdot x}
   \end{align}$$

 - In :func:`~kszx.simulate_gaussian_field()` and :func:`~kszx.estimate_power_spectrum()`, we use
   the following normalization for the power spectrum $P(k)$:

   $$\langle f(k) f(k')^* \rangle = V_{\rm box} P(k) \delta_{kk'}$$

 - Idea behind these conventions: in a finite pixelized box, these conventions are as similar
   as possible to the following infinite-volume continuous conventions:

   $$\begin{align}
   f(k) &= \int d^nx\, f(x) e^{-ik\cdot x} \\
   f(x) &= \int \frac{d^nk}{(2\pi)^n} \, f(k) e^{ik\cdot x} \\
   \langle f(k) f(k')^* \rangle &= P(k) (2\pi)^n \delta^n(k-k')
   \end{align}$$

FFTs with nonzero "spin"
^^^^^^^^^^^^^^^^^^^^^^^^

 - We define spin-1 Fourier transforms by inserting an extra factor
   $(\pm i {\hat k} \cdot {\hat r})$:

   $$\begin{align}
   f(k) &= V_{pix} \sum_x f(x) (-i {\hat k} \cdot {\hat r}) e^{-ik\cdot x} \\
   f(x) &= V_{box}^{-1} \sum_k f(k) (i {\hat k} \cdot {\hat r}) e^{ik\cdot x}
   \end{align}$$

   where the line-of-sight direction $\hat r$ is defined in "observer coordinates"
   (see :class:`~kszx.Box` for more info).

 - Application: ``kszx.fft_c2r(..., spin=1)`` can be used to compute the radial velocity
   field from the density field (with a factor $faH/k$). In equations, we have (in a
   constant-time "snapshot" without lightcone evolution):

   $$v_r(x) = \int \frac{d^3k}{(2\pi)^3} \, \frac{faH}{k} (i {\hat k} \cdot {\hat r}) \delta_m(k) e^{ik\cdot x}$$
   
   Code might look like this::

    box = kszx.Box(...)
    cosmo = kszx.Cosmology('planck18+bao')

    # delta_m = Fourier-space density field at z=0
    delta_m = kszx.simulate_gaussian_field(box, lambda k: cosmo.Plin_z0(k))

    # vr = Real-space radial velocity field at z=0
    f = cosmo.frsd(z=0)
    H = cosmo.H(z=0)
    vr = kszx.multiply_kfunc(box, delta, lambda k: f*H/k, dc=0)
    vr = kszx.fft_c2r(box, vr, spin=1)

 - Another application: the spin-1 r2c transform can be used to estimate $P_{gv}(k)$
   or $P_{vv}(k)$ from the radial velocity field (or the kSZ velocity reconstruction),
   by calling :func:`~kszx.fft_r2c()` followed by :func:`~kszx.estimate_power_spectrum()`.
   
   Code might look like this::

     box = kszx.Box(...)
     kbin_edges = np.linspace(0, 0.1, 11)   # kmax=0.1, nkbins=10

     # Assume we have real-space maps delta_g (galaxy field) and vr (kSZ velocity reconstruction)
     delta_g = ....  # real-space map (dtype float, shape box.real_space_shape)
     vr = ...        # real-space map (dtype float, shape box.real_space_shape)

     # Real space -> Fourier space
     delta_g = kszx.fft_r2c(box, delta_g)   # spin=0 is the default
     vr = kszx.fft_r2c(box, vr, spin=1)     # note spin=1 for radial velocity!

     # Returns a shape (2,2,nkbins) array, containing P_{gg}, P_{g,vr}, P_{vr,vr}.
     # Note that power spectra are unnormalized -- see estimate_power_spectrum() docstring.
     pk = kszx.estimate_power_spectrum(box, [delta_g,vr], kbin_edges)
     
   (In a real pipeline, you'd probably use the high-level class :class`~kszx.KszPSE`,
   which has "bells and whistles" such as power spectrum normalization. This example
   code is intended to illustrate low-level building blocks: :func:`~kszx.fft_r2c()`
   and :func:`~kszx.estimate_power_spectrum()`.)
     
 - At some point in the future, I'll define spin-$l$ transforms, with an
   extra factor $(\pm i^l P_l({\hat k} \cdot {\hat r}))$. For example, spin-2
   transforms can be used to simulate/estimate "quadrupolar" RSDs, analogously
   to the "dipolar" radial velocity field from the previous two bullet points.

 - In addition to :func:`~kszx.fft_r2c()` and :func:`~kszx.fft_c2r()`, the ``spin``
   optional argument also occurs in scattered functions which include an FFT step.
   For example, :func:`~kszx.grid_points()` with ``fft=True``.
   
   
