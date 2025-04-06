:mod:`CatalogGridder`
=====================
		
.. autoclass:: kszx.CatalogGridder
       
   .. automethod:: kszx.CatalogGridder.grid_density_field
   .. automethod:: kszx.CatalogGridder.grid_sampled_field
   .. automethod:: kszx.CatalogGridder.compute_A

.. _catalog_gridder_details:

Details
-------

In this appendix, we explain what CatalogGridder actualy computes.

The estimator $\hat P_{ff'}^{\rm nrand}(k)$
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :class:`~kszx.CatalogGridder` is constructed, each footprint is specified by points
$x_j^{rand}$ and ``rweights`` $W_j^{rand}$. Let $R(x)$ be the "footprint field" defined by:

$$R(x) \equiv \sum_{j\in rand} W_j^{rand} \delta^3(x-x_j)$$

In this section, we define a power spectrum estimator $\hat P_{ff'}^{\rm nrand}(k)$ whose
input is a pair of Fourier-space fields $f,f'$. The estimator $\hat P_{ff'}^{\rm nrand}(k)$
is "normalized to randoms" in the following sense. Suppose that the fields $f,f'$ are
obtained by multiplying "unwindowed" fields $F,F'$ by the footprint fields $R,R'$:

$$\begin{align}
f(x) &= R(x) F(x) = \sum_{j\in rand} W_j F(x_j) \delta^3(x-x_j) \\
f'(x) &= R'(x) F'(x) = \sum_{j\in rand'} W'_j F(x'_j) \delta^3(x-x'_j)
\end{align}$$

Then, the estimator $P^{\rm nrand}_{ff'}(k)$ has (to a good approximation) the same
normalization as the unwindowed power spectrum $P_{FF'}(k)$.

Let $P^{raw}_{RR'}(k)$ be the (unnormalized) power spectrum in volume $V_{box}$, and define:

$$A_{RR'} \equiv \left(\int_{k < 2^{1/3}K} - \int_{2^{1/3}K < k < K} \right) \frac{d^3k}{(2\pi)^3} P^{raw}_{RR'}(k)$$

The purpose of the subtraction is to cancel shot noise.
The value of $A_{RR'}$ should be roughly independent of the choice of $K$.
We choose $K = 0.6 k_{\rm nyq}$ (I didn't put much thought into this).

To get some intuition for what $A_{RR'}$ represents, suppose all weights have
constant values $W,W'$, and the randoms have number densities $n,n'$ in volumes $V,V'$. Then:

$$A_{RR'} \approx nn'WW' \frac{V \cap V'}{V_{\rm box}}$$

We define $P^{\rm nrand}_{ff'}(k)$ by:

$$P^{\rm nrand}_{ff'}(k) = N_{RR'} \, P^{raw}_{ff'}(k)$$

where the matrix $N_{RR'}$ (stored in ``CatalogGridder.ps_normalization``) is defined by:

$$N_{RR'} \equiv \begin{cases}
1/A_{RR'} & \mbox{ if } A_{RR'}^2 \ge A_{RR} A_{R'R'} \\
0 & \mbox{ otherwise}
\end{cases}$$

Density fields
^^^^^^^^^^^^^^

When :meth:`~kszx.CatalogGridder.grid_density_field()` is called, the caller specifies
points $x_i$ and ``gweights`` $W_i^{\rm gal}$. We define the density field $\delta(x)$ by:

$$\delta(x) \equiv \left( \frac{W_{\rm tot}^{\rm rand}}{W_{\rm tot}^{\rm gal}} \sum_{i\in\rm gal} W_i^{\rm gal} \delta^3(x-x_i) \right) - R(x)$$

where $W_{\rm tot}^{\rm gal} \equiv \sum_i W_i^{\rm gal}$, $W_{\rm tot}^{\rm rand} \equiv \sum_j W_j^{\rm rand}$, and $R(x)$ was defined above.

This normalization of $\delta(x)$ is nonstandard, but if we apply $P_{\delta\delta}^{\rm nrand}$, we end up with the correct
normalization for clustering power spectra for an overdensity field $\delta_g$.

Sampled fields
^^^^^^^^^^^^^^

When :meth:`~kszx.CatalogGridder.grid_sampled_field()` is called, the caller specifies
points $x_i$ and ``coeffs`` $c_i$. The idea is that each coefficient $c_i$ traces some
underlying continuous field $F(x)$ with weight $w_i^{\rm gal}$:

$$c_i = w_i^{\rm gal} F(x_i)$$

We define the sampled field $f(x)$ by:

$$f(x) = \frac{w_{\rm tot}^{\rm rand}}{w_{\rm tot}^{\rm gal}} \sum_{i\in \rm gal} C_i \delta^3(x-x_i)$$

where $w_{\rm tot}^{\rm gal} \equiv \sum_i w_i^{\rm gal}$ and $w_{\rm tot}^{\rm rand} \equiv \sum_j w_j^{\rm rand}$.

This normalization of $f(x)$ has the property that if we apply $P_{ff'}^{\rm nrand}$, we end up with the correct
normalization for clustering power spectra for the underlying continuous field $F(x)$. Note that in order to normalize,
we don't need the individual sampling weights $w_i^{\rm gal}$, only their sum $w_{\rm tot}^{\rm gal} = \sum_i w_i^{\rm gal}$.


