:mod:`Window functions`
=======================

.. automodule:: kszx.wfunc_utils

.. autofunction:: kszx.wfunc_utils.compute_wcrude
.. autofunction:: kszx.wfunc_utils.compare_pk

.. _wcrude_details:

Details of $W_{crude}$
----------------------

In this appendix, we explain what :func:`kszx.wfunc_utils.compute_wcrude()` actually computes.

Let $R(x)$, $R'(x)$ be "footprint fields", and let $P_{RR'}^{raw}(k)$ be their unnormalized cross
power spectrum. We define:

$$W_{RR'} \equiv \left(\int_{k < 2^{1/3}K} - \int_{2^{1/3}K < k < K} \right) \frac{d^3k}{(2\pi)^3} P^{raw}_{RR'}(k)$$

The purpose of the subtraction is to cancel shot noise.
The value of $W_{RR'}$ should be roughly independent of the choice of $K$.
We choose $K = 0.6 k_{\rm nyq}$ (I didn't put much thought into this).

To get some intuition for what $W_{RR'}$ represents, suppose that footprint field $R(x)$
is defined by summing over randoms with number density $n$ in volume $V$, with constant
weight $w$:

$$R(x) = \sum_{j\in rand} w \delta^3(x-x_j)$$

and similarly for footprint field $R'(x)$, with $(n,V,w) \rightarrow (n',V',w')$.
Then:

$$W_{RR'} \approx nn'WW' \frac{V \cap V'}{V_{\rm box}}$$

