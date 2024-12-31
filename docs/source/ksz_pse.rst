:mod:`KszPSE`
=============

.. autoclass:: kszx.KszPSE
    :members:

.. _ksz_pse_details:

KszPSE details
--------------

Galaxy density field $\rho_g$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Placeholder.

KSZ velocity reconstruction $\hat v_r$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Placeholder.

Surrogate galaxy field $S_g$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall the following notation from the main overleaf:
$$\begin{align}
W_L(x) &= \mbox{Galaxy weight function (e.g.~FKP)}  \\
\rho_g(x) &= \bigg( \sum_{i\in \rm gal} W_i^L \, \delta^3(x-x_i) \bigg) - \frac{N_g}{N_r} \bigg( \sum_{j\in \rm rand} W_j^L \, \delta^3(x-x_j) \bigg) \\
S_g^{\rm sig}(x) &= \frac{N_g}{N_r} \sum_{j\in \rm rand} W_j^L \, \delta_G(x_j) \, \delta^3(x-x_j) \\
\delta_G(k,z) &= \left( b_g(z) + \frac{f_{NL}}{\alpha(k,z)} b_{ng}(z) \right) \delta_{\rm lin}(k,z) \\
S_g^{\rm noise}(x) &= \frac{N_g}{N_r} \sum_{j\in \rm rand} W_j^L \,  \eta_j \, \delta^3(x-x_j) \\
\big\langle \eta_j^2 \big\rangle &= \frac{N_r}{N_g} - \big\langle \delta_G(z_j)^2 \big\rangle
\end{align}$$
These definitions are arranged so that $S_g(x) \equiv (S_g^{\rm sig}(x) + S_g^{\rm noise}(x))$
has the same power spectrum as $\rho_g(x)$.

Surrogate radial velocity field $S_v$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall the following notation from the main overleaf:
$$\begin{align}
W_S(x) &= \mbox{Galaxy weight function (e.g.~FKP)}  \\
\hat v_r(x) &= \sum_{i\in \rm gal} W_i^S \, \tilde T(\theta_i) \, \delta^3(x-x_i)  \\
S_v^{\rm sig}(x) &= \frac{N_g}{N_r} \sum_{j\in\rm rand} W_j^S b_j^v \, v_r(x_j) \, \delta^3(x-x_j)  \\
S_v^{\rm noise}(x) &= \sum_{j\in\rm rand} M_j W_j^S \tilde T(\theta_j) \, \delta^3(x-x_j)  \\
M_j &= \begin{cases}
1 & \mbox{if $j$ is in a randomly selected subset of size } N_{\rm gal} \\ 
0 & \mbox{otherwise}
\end{cases}
\end{align}$$
These definitions are arranged so that $S_v(x) \equiv (S_v^{\rm sig}(x) + S_v^{\rm noise}(x))$
has the same power spectrum as $\hat v_r(x)$.

Interface with CatalogGridder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Placeholder.
