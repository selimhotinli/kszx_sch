:mod:`KszPSE`
=============

.. autoclass:: kszx.KszPSE
    :members:

.. _ksz_pse_details:

KszPSE details
--------------

In this appendix, we explain what KszPSE actually computes.

Galaxy density field $\rho_g$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall the following notation from the main overleaf:
$$\begin{align}
W_L(x) &= \mbox{Large-scale galaxy weight function (e.g. FKP)}  \\
\rho_g(x) &= \bigg( \sum_{i\in \rm gal} W_i^L \, \delta^3(x-x_i) \bigg) - \frac{N_g}{N_r} \bigg( \sum_{j\in \rm rand} W_j^L \, \delta^3(x-x_j) \bigg)
\end{align}$$
In the code, a galaxy field $\rho_g(x)$ is created when  :meth:`~kszx.KszPSE.eval_pk()` is called.

 - The galaxy locations $x_i$ are taken from the ``gcat`` argument.
 - The large-scale weights $W_i^L$ are taken from the ``gweights`` argument.
 - The corresponding quantities $(x_j, W_j^L)$ for the randoms were specified at construction (``rcat`` and ``rweights`` constructor args).
 - To compute $\rho_g(x)$, we call :meth:`kszx.CatalogGridder.grid_density_field()`.

KSZ velocity reconstruction $\hat v_r$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall the following notation from the main overleaf:
$$\begin{align}
W_S(x) &= \mbox{Small-scale galaxy weight function (e.g. FKP)}  \\
\hat v_r(x) &= \sum_{i\in \rm gal} W_i^S \, \tilde T(\theta_i) \, \delta^3(x-x_i)
\end{align}$$
In the code, a velocity reconstruction field $\hat v_r(x)$ is created when  :meth:`~kszx.KszPSE.eval_pk()` is called.

 - The galaxy locations $x_i$ are taken from the ``gcat`` argument.
 - The small-scale weights $W_i^S$ are taken from the ``ksz_gweights`` argument.
 - The filtered CMB temperature $\tilde T(\theta_i)$ is taken from the ``ksz_tcmb`` argument.
 - The ``ksz_bv`` argument gives the biasing relation $\tilde T(\theta_i) = b_v^i v_r(x_i) + (\mbox{noise})$.
   
To compute $\hat v_r(x)$, we call :meth:`kszx.CatalogGridder.grid_sampled_field()`, with:

 - ``coeffs`` argument given by $c_i = W_i^S \tilde T(\theta_i)$
 - ``wsum`` argument given by $\sum W_i^S b_i^v$.

To see this, we note that the previous chain of equations can be written:
$$\hat v_r(x) = \sum_i c_i \delta^3(x-x_i) \hspace{1.5cm} c_i = W_i^S \tilde T(\theta_i) = W_i^S b_i^v v_r(x_i) + (\mbox{noise})$$
and compare with the :meth:`kszx.CatalogGridder.grid_sampled_field()` docstring.

Surrogate galaxy field $S_g$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall the following notation from the main overleaf:
$$\begin{align}
S_g(x) &= \sum_{j\in \rm rand} (S_j^{\rm sig} + S_j^{\rm noise}) \delta^3(x-x_j) \\
S_j^{\rm sig} &= \frac{N_g}{N_r} W_j^L \, \delta_G(x_j) \\
S_j^{\rm noise} &= \frac{N_g}{N_r} W_j^L \eta_j \\
\delta_G(k,z) &= \left( b_g(z) + \frac{f_{NL}}{\alpha(k,z)} b_{ng}(z) \right) \delta_{\rm lin}(k,z) \\
\big\langle \eta_j^2 \big\rangle &= \frac{N_r}{N_g} - \big\langle \delta_G(z_j)^2 \big\rangle
\end{align}$$
These definitions are arranged so that $S_g(x)$ has the same power spectrum as $\rho_g(x)$.

In the code, the coefficients $(S_j^{\rm sig} + S_j^{\rm noise})$ are simulated when :meth:`~kszx.KszPSE.simulate_surrogate()`
is called, and stored in ``self.Sg_coeffs`` and ``self.dSg_dfNL``.
All data needed to simulate these coefficients is specified at construction:

  - The locations $x_j$ are taken from the ``rcat`` constructor arg.
  - The large-scale weights $W_j^L$ are taken from the ``rweights`` constructor arg. 
  - The number of galaxies $N_g$ is a Gaussian random variable, with mean/rms given by the ``surr_ngal_mean``, ``surr_ngal_rms`` constructor args.
  - The galaxy bias $b_g$ is given by the ``surr_bg`` constructor arg.

The surrogate field $S_g(x)$ is computed from the coefficients $(S_j^{\rm sig} + S_j^{\rm noise})$ when
:meth:`~kszx.KszPSE.eval_pk_surrogate()` is called.
To compute $S_g(x)$, we call :meth:`kszx.CatalogGridder.grid_sampled_field()`, with:

  - ``coeffs`` argument given by $(S_j^{\rm sig} + S_j^{\rm noise})$
  - ``wsum`` argument given by $(N_g/N_r)  \sum_j W_j^L$.

To see this, we note that the previous chain of equations can be written:
$$\hat S_g(x) = \sum_j c_j \delta^3(x-x_j) \hspace{1.5cm} c_j = (S_j^{\rm sig} + S_j^{\rm noise}) = \frac{N_g}{N_r} W_j^L \delta_G(x_j) + (\mbox{noise})$$
and compare with the :meth:`kszx.CatalogGridder.grid_sampled_field()` docstring.

Surrogate radial velocity field $S_v$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall the following notation from the main overleaf:
$$\begin{align}
S_v(x) &= \sum_{j\in\rm rand} (S_j^{\rm sig} + S_j^{\rm noise}) \delta^3(x-x_j) \\
S_j^{\rm sig} &= \frac{N_g}{N_r} W_j^S b_j^v \, v_r(x_j) \\
S_j^{\rm noise} &= M_j W_j^S \tilde T(\theta_j)   \\
M_j &= \begin{cases}
1 & \mbox{if $j$ is in a randomly selected subset of size } N_{\rm gal} \\ 
0 & \mbox{otherwise}
\end{cases}
\end{align}$$
These definitions are arranged so that $S_v(x)$ has the same power spectrum as $\hat v_r(x)$.

When :meth:`~kszx.KszPSE.simulate_surrogate()` is called, the coefficients $S_j^{\rm sig}$ and $S_j^{\rm noise})$ are simulated 
and stored in ``self.Sv_signal``, ``self.Sv_noise``.
All data needed to simulate these coefficients is specified at construction:

  - The locations $x_j$ are taken from the ``rcat`` constructor arg.
  - The small-scale weights $W_j^S$ are taken from the ``ksz_rweights`` constructor arg. 
  - The number of galaxies $N_g$ is a Gaussian random variable, with mean/rms given by the ``surr_ngal_mean``, ``surr_ngal_rms`` constructor args.
  - The per-object kSZ velocity reconstruction bias is given by the constructor arg ``ksz_bv``.
  - The CMB realization $\tilde T(\theta_j)$ used to "bootstrap" the reconstruction noise is given by the ``ksz_tcmb_realization`` constructor arg.

When :meth:`~kszx.KszPSE.eval_pk_surrogate()` is called, surrogate fields $S_v(x)$ are simualted.
To compute $S_v(x)$, we call :meth:`kszx.CatalogGridder.grid_sampled_field()`, with:

  - ``coeffs`` argument given by $S_j^{\rm sig}$ or $S_j^{\rm noise}$
  - ``wsum`` argument given by $(N_g/N_r) \sum_j W_j^S b_j^v$.

To see this, we note that the previous chain of equations can be written:
$$\hat S_v(x) = \sum_j c_j \delta^3(x-x_j) \hspace{1.5cm} c_j = (S_j^{\rm sig} + S_j^{\rm noise}) = \frac{N_g}{N_r} W_j^S b_j^v v_r(x_i) + (\mbox{noise})$$
and compare with the :meth:`kszx.CatalogGridder.grid_sampled_field()` docstring.
