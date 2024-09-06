#include <iostream>
#include "cpp_kernels.hpp"

using namespace std;

using cplx = std::complex<double>;


inline double square(double x)
{
    return x*x;
}

inline double mult_zz(cplx x, cplx y)
{
    return x.real() * y.real() + x.imag() * y.imag();
}


struct pse_args
{
    double        *out_pk;       // length nkbins * M^2, where M = nmaps
    long          *out_bcounts;  // length nkbins
    int            nkbins;
    const double  *k_delim;      // length (nkbins+1)
};


// -------------------------------------------------------------------------------------------------
//
// Power spectrum estimation (pse) kernels:
// The template parameter M is the number of maps.
// 
//    inline void pse_1d<M> (const pse_args &args, double curr_k2, int curr_bin,
//                           int np, double kf, const cplx **maps, const int *strides);
//
//    void pse<M> (const pse_args &args, double curr_k2, int curr_bin, int ndim,
//                 const int *np, const double *kf, const cplx **maps, const int *strides);
//
// Args:
//
//   - maps: shape=(ndim,M) dtype=(const cplx *)
//
//       The PSE kernel assumes that the first M entries have been populated,
//       and the remaning (ndim-1)*M entries can be used as scratch space
//       (for recursive calls to pse<M> with ndim -> (ndim-1)).
//
//   - strides: shape=(ndim,M) dtype=int
//
//       Note that spatial dimension is the slowly varying index, and map
//       index is rapidly varying.
//
// Caller must ensure that 'curr_bin' satisfies the following constraints:
//
//    - Constraint 1:  (-1) <= curr_bin < args.nkbins
//
//    - Constraint 2: bin_kmin^2 <= curr_k2 < bin_kmax^2
//
//         where bin_kmin = k_delim[curr_bin] if (curr_bin >= 0) else 0
//               bin_kmax = k_delim[curr_bin+1]
// 
// Notes:
//
//    - If curr_k2 >= k_delim[nkbins]^2, then you should skip the call to pse(),
//      since there is no way to satisfy the above constraints.
//
//    - Note: pse<M> () is recursive, and therefore non-inline.
//
//    - There's currently a lot of cut-and-paste between kernels -- I may clean
//      up later with inline functions and template magic.
//
//    - Note: currently limited to Mmax=2!
//
//      My long-term plan is to write more pse_1d<M> kernels, say up to Mmax ~ 8,
//      and then write a non-templated "highM" kernel to handle M > Mmax.
//
//      NOTE!! If the C++ code is modified to allow higher Mmax, make sure to
//      update the unit test (test_estimate_power_spectrum()).


template<int M>
inline void pse_1d(const pse_args &args, double curr_k2, int curr_bin, int np, double kf, const cplx **maps, const int *strides);


template<> inline void pse_1d<1> (const pse_args &args, double curr_k2, int curr_bin, int np, double kf, const cplx **maps, const int *strides)
{
    double *out_pk = args.out_pk;
    long *out_bcounts = args.out_bcounts;
    
    int nkbins = args.nkbins;
    const double *k_delim = args.k_delim;

    const cplx *map = maps[0];
    int s = strides[0];

    for (int ik = 0; 2*ik <= np; ik++) {
	// Update curr_bin.
	double k2 = curr_k2 + square(ik*kf);
	while (k2 >= square(k_delim[curr_bin+1]))
	    if (++curr_bin == nkbins)
		return;
	
	int m = ((ik==0) || (2*ik==np)) ? 1 : 2;
	cplx z = map[0];
	map += s;
	
	if (curr_bin < 0)
	    continue;

	out_pk[curr_bin] += m * mult_zz(z, z);
	out_bcounts[curr_bin] += m;
    }
}


template<> inline void pse_1d<2> (const pse_args &args, double curr_k2, int curr_bin, int np, double kf, const cplx **maps, const int *strides)
{
    double *out_pk = args.out_pk;
    long *out_bcounts = args.out_bcounts;
    
    int nkbins = args.nkbins;
    const double *k_delim = args.k_delim;

    const cplx *map0 = maps[0];
    const cplx *map1 = maps[1];
    int s0 = strides[0];
    int s1 = strides[1];

    for (int ik = 0; 2*ik <= np; ik++) {
	// Update curr_bin.
	double k2 = curr_k2 + square(ik*kf);
	while (k2 >= square(k_delim[curr_bin+1]))
	    if (++curr_bin == nkbins)
		return;

	int m = ((ik==0) || (2*ik==np)) ? 1 : 2;
	cplx z0 = map0[0];
	cplx z1 = map1[0];
	map0 += s0;
	map1 += s1;
	
	if (curr_bin < 0)
	    continue;

	out_pk[3*curr_bin] += m * mult_zz(z0,z0);
	out_pk[3*curr_bin+1] += m * mult_zz(z0,z1);
	out_pk[3*curr_bin+2] += m * mult_zz(z1,z1);
	out_bcounts[curr_bin] += m;
    }
}


template<int M>
void pse(const pse_args &args, double curr_k2, int curr_bin, int ndim, const int *np, const double *kf, const cplx **maps, const int *strides)
{
    int np0 = np[0];
    double kf0 = kf[0];

    if (ndim == 1) {
	pse_1d<M> (args, curr_k2, curr_bin, np0, kf0, maps, strides);
	return;
    }
    
    int nkbins = args.nkbins;
    const double *k_delim = args.k_delim;
    
    for (int m = 0; m < M; m++)
	maps[M+m] = maps[m];
    
    for (int ik0 = 0; ik0 < np0; ik0++) {
	int ik = std::min(ik0, np0-ik0);;
	double k2 = curr_k2 + square(ik*kf0);

	// Update curr_bin
	while ((curr_bin >= 0) && (k2 < square(k_delim[curr_bin])))
	    curr_bin--;   // search down
	while ((curr_bin < nkbins) && (k2 >= square(k_delim[curr_bin+1])))
	    curr_bin++;   // search up

	// Call pse<M>() recursively, with ndim -> (ndim-1).
	if (curr_bin < nkbins)
	    pse<M> (args, k2, curr_bin, ndim-1, np+1, kf+1, maps+M, strides+M);

	for (int m = 0; m < M; m++)
	    maps[m+M] += strides[m];
    }
}


// -------------------------------------------------------------------------------------------------


py::tuple estimate_power_spectrum(py::list map_list, py::array_t<const double> &k_delim, py::array_t<const long> &npix, py::array_t<const double> &kf, double box_volume)
{
    if (k_delim.ndim() != 1)
	throw runtime_error("estimate_power_spectrum: expected k_delim.ndim == 1");
    if (npix.ndim() != 1)
	throw runtime_error("estimate_power_spectrum: expected npix.ndim == 1");
    if (kf.ndim() != 1)
	throw runtime_error("estimate_power_spectrum: expected kf.ndim == 1");
    if (npix.shape(0) != kf.shape(0))
	throw runtime_error("estimate_power_spectrum: expected len(npix) == len(kf)");
    if (box_volume <= 0)
	throw runtime_error("estimate_power_spectrum: expected box_volume > 0");
    
    if (get_int_stride(k_delim,0) != 1)
	throw runtime_error("estimate_power_spectrum: expected k_delim to be contiguous array");
    if (get_int_stride(npix,0) != 1)
	throw runtime_error("estimate_power_spectrum: expected npix to be contiguous array");
    if (get_int_stride(kf,0) != 1)
	throw runtime_error("estimate_power_spectrum: expected kf to be contiguous array");
    
    int nmaps = map_list.size();
    int nkbins = get_int_shape(k_delim,0) - 1;    // note -1 here
    int ndim = get_int_shape(npix,0);

    if (ndim < 1)
	throw runtime_error("estimate_power_spectrum: expected len(npix) >= 1");
    if (nkbins < 1)
	throw runtime_error("estimate_power_spectrum: expected len(k_delim) >= 2");
    if (nmaps < 1)
	throw runtime_error("estimate_power_spectrum: expected map_list to be a nonempty list");
    if (nmaps > 2)
	throw runtime_error("estimate_power_spectrum: We currently only support nmaps >= 2!"
			    " This is easy to change, but requires minor modifications to the C++ code");

    const double *kdd = k_delim.data();
    if (kdd[0] < 0.)
	throw runtime_error("estimate_power_spectrum: expected k_delim[0] >= 0");

    for (int b = 0; b < nkbins; b++)
	if (kdd[b] >= kdd[b+1])
	    throw runtime_error("estimate_power_spectrum: expected k_delim array to be sorted");

    // Allocate some arrays needed by the PSE kernel.
    // Note: 'map_vec' is length (ndim * nmaps), but only the first nmaps entries will be
    // initialized here. (Remaning entries are used as scratch space by the PSE kernel.)
    
    vector<int> np(ndim);
    vector<const cplx *> map_vec(ndim * nmaps, nullptr);
    vector<int> strides(ndim * nmaps);  // spatial dimension is major index, map is minor index

    // Error-check 'npix' argument, and populate 'np' vector (converting long -> int).
    
    for (int axis = 0; axis < ndim; axis++) {
	long n = npix.data()[axis];
	if (n <= 0)
	    throw runtime_error("estimate_power_spectrum: expected npix > 0");
	if (n > INT_MAX)
	    throw runtime_error("estimate_power_spectrum: npix >= 2^31?!");
	np[axis] = n;
    }

    // Error-check 'map_list' argument, and populate vectors 'map_vec', 'strides'.
    
    for (int imap = 0; imap < nmaps; imap++) {
	py::object obj = map_list[imap];
	py::array_t<complex<double>> arr = obj;

	if (arr.ndim() != ndim)
	    throw runtime_error("estimate_power_spectrum: expected map.ndim == len(npix)");
	
	map_vec[imap] = arr.data();
	
	for (int axis = 0; axis < ndim; axis++) {
	    int n_expected = (axis < (ndim-1)) ? np[axis] : ((np[axis]/2) + 1);
	    if (arr.shape(axis) != n_expected)
		throw runtime_error("estimate_power_spectrum: map shape is inconsistent with 'npix'");
	    
	    strides[axis*nmaps + imap] = get_int_stride(arr, axis);
	}
    }

    // Run PSE kernel.

    int M2 = ((nmaps) * (nmaps+1)) / 2;
    vector<double> tmp_pk(nkbins * M2);
    py::array_t<long> ret_bcounts({nkbins});

    pse_args args;
    args.out_pk = &tmp_pk[0];
    args.out_bcounts = ret_bcounts.mutable_data();
    args.nkbins = nkbins;
    args.k_delim = k_delim.data();  // contiguous (checked above)

    memset(args.out_pk, 0, nkbins * nmaps * nmaps * sizeof(*args.out_pk));
    memset(args.out_bcounts, 0, nkbins * sizeof(*args.out_bcounts));
    int curr_bin = (args.k_delim[0] > 0.) ? -1 : 0;

    if (nmaps == 1)
	pse<1> (args, 0.0, curr_bin, ndim, &np[0], kf.data(), &map_vec[0], &strides[0]);
    else if (nmaps == 2)
	pse<2> (args, 0.0, curr_bin, ndim, &np[0], kf.data(), &map_vec[0], &strides[0]);
    else
	throw runtime_error("estimate_power_spectrum: unsupported 'nmaps'");

    // Copy tmp_pk -> ret_pk (array orderings are different), and apply normalization.
    
    vector<double> pk_norm(nkbins);
    py::array_t<double> ret_pk({nmaps,nmaps,nkbins});
    double *rptr = ret_pk.mutable_data();
    
    for (int b = 0; b < nkbins; b++) {
	long n = args.out_bcounts[b];
	pk_norm[b] = (n > 0) ? (1.0 / double(n) / box_volume) : 0.0;
    }
    
    int isrc = 0;  // represents index pair (m,m2)
    for (int m = 0; m < nmaps; m++) {
	for (int m2 = 0; m2 <= m; m2++) {
	    int idst1 = (m*nmaps + m2) * nkbins;
	    int idst2 = (m2*nmaps + m) * nkbins;
	    
	    for (int b = 0; b < nkbins; b++) {
		double pk = pk_norm[b] * tmp_pk[b*M2 + isrc];
		rptr[idst1+b] = pk;
		rptr[idst2+b] = pk;
	    }
	    
	    isrc++;
	}
    }

    return py::make_tuple(ret_pk, ret_bcounts);
}
