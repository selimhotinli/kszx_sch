#include "cpp_kernels.hpp"

using namespace std;


// -------------------------------------------------------------------------------------------------
//
// interpolation_args: this helper class handles the arguments (grid, points, lpos, pixsize)
// that are common to all interpolation/gridding kernels.
//
// FIXME currently assumes ndim=3 and "width-2" kernel (e.g. CIC).


// Use T=(const double) for interpolation, and T=(double) for gridding.
template<typename T>
struct interpolation_args
{
    T *gdata;           // grid data
    int gn0, gn1, gn2;  // grid shape
    int gs0, gs1, gs2;  // grid strides

    const double *pdata;  // points data
    long npoints;         // points array has shape (npoints, ndim)
    int ps0, ps1;         // strides in 2-d points array

    // Used to translate 'points' to grid_coords in get_xyz().
    double lpos0, lpos1, lpos2, rec_ps, rec_pvol;

    // Optional: weights array
    const double *wdata = nullptr;
    int ws = 0;   // stride in 1-d weights array
    

    // This constructor does not have a 'weights' array, and is used in interpolation kernels.
    interpolation_args(py::array_t<T> &grid, py::array_t<const double> &points, double lpos0_, double lpos1_, double lpos2_, double pixsize)
    {
	// Argument checking.
    
	if (grid.ndim() != 3)
	    throw runtime_error("expected 'grid' to be a 3-d array");
	if (points.ndim() != 2)
	    throw runtime_error("expected 'points' to be a 2-d array");
	if (points.shape(1) != 3)
	    throw runtime_error("expected 'points' to be a shape (N,3) array");
	if (pixsize <= 0)
	    throw runtime_error("expected pixsize > 0");

	// Unpack python arrays into C++ pointers/ints.

	if constexpr (std::is_const<T>::value)
            gdata = grid.data();
        else
            gdata = grid.mutable_data();
	
	gn0 = get_int_shape(grid, 0);
	gn1 = get_int_shape(grid, 1);
	gn2 = get_int_shape(grid, 2);
	gs0 = get_int_stride(grid, 0);
	gs1 = get_int_stride(grid, 1);
	gs2 = get_int_stride(grid, 2);

	if ((gn0 < 2) || (gn1 < 2) || (gn2 < 2))
	    throw runtime_error("expected all grid dimensions >= 2");
	
	pdata = points.data();
	npoints = points.shape(0);
	ps0 = get_int_stride(points, 0);
	ps1 = get_int_stride(points, 1);

	lpos0 = lpos0_;
	lpos1 = lpos1_;
	lpos2 = lpos2_;
	rec_ps = 1.0 / pixsize;
	rec_pvol = rec_ps * rec_ps * rec_ps;
    }

    // This constructor does have a 'weights' array, and is used in gridding kernels.
    interpolation_args(py::array_t<T> &grid, py::array_t<const double> &points, py::array_t<const double> &weights, double lpos0_, double lpos1_, double lpos2_, double pixsize)
	: interpolation_args(grid, points, lpos0_, lpos1_, lpos2_, pixsize)
    {
	if (weights.ndim() != 1)
	    throw runtime_error("expected 'weights' to be a 1-d array");
	if (weights.shape(0) != npoints)
	    throw runtime_error("expected 'weights' to be a length-N array, where N=points.shape[0]");

	wdata = weights.data();
	ws = get_int_stride(weights, 0);
    }

    
    // Get (x,y,z) in "grid coordinates".
    inline void get_xyz(long i, double &x, double &y, double &z)
    {
	x = rec_ps * (pdata[i*ps0] - lpos0);
	y = rec_ps * (pdata[i*ps0 + ps1] - lpos1);
	z = rec_ps * (pdata[i*ps0 + 2*ps1] - lpos2);
    }
    
    // Get (x,y,z) in "grid coordinates", plus associated gridding weight.
    inline void get_xyzw(long i, double &x, double &y, double &z, double &w)
    {
	get_xyz(i, x, y, z);
	w = wdata[i*ws] * rec_pvol;  // note factor rec_pvol = 1 / (pixel volume)
    }
};


// -------------------------------------------------------------------------------------------------


struct cic_axis
{
    int stride;
    double w0, w1;
};


// The template lets 'arr' be either (double *) or (const double *).
template<typename T>
inline T *cic_axis_init(cic_axis &ax, T *arr, double x, int n, int stride)
{
    int i = int(x);
    bool bad = (i >= n-1) || (i < 0) || (x < 0.0);

    if (_unlikely(bad))
	throw runtime_error("kszx.cpp_kernels.cic_interpolate_3d(): point is out of bounds");

    double dx = x - (double)i;
    ax.stride = stride;
    ax.w0 = 1.0 - dx;
    ax.w1 = dx;
    
    return arr + long(i) * long(stride);
}


// -------------------------------------------------------------------------------------------------
//
// CIC interpolation.


inline double cic_interp_1d(cic_axis &ax, const double *arr)
{
    return (ax.w0 * arr[0]) + (ax.w1 * arr[ax.stride]);
}

inline double cic_interp_2d(cic_axis &ax0, cic_axis &ax1, const double *arr)
{
    double f0 = cic_interp_1d(ax1, arr);
    double f1 = cic_interp_1d(ax1, arr + ax0.stride);
    return (ax0.w0 * f0) + (ax0.w1 * f1);
}

inline double cic_interp_3d(cic_axis &ax0, cic_axis &ax1, cic_axis &ax2, const double *arr)
{
    double f0 = cic_interp_2d(ax1, ax2, arr);
    double f1 = cic_interp_2d(ax1, ax2, arr + ax0.stride);
    return (ax0.w0 * f0) + (ax0.w1 * f1);
}


py::array_t<double> cic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points, double lpos0, double lpos1, double lpos2, double pixsize)
{
    interpolation_args<const double> args(grid, points, lpos0, lpos1, lpos2, pixsize);

    py::array_t<double> ret({args.npoints});
    double *rdata = ret.mutable_data();

    for (long i = 0; i < args.npoints; i++) {
	double x, y, z;
	args.get_xyz(i, x, y, z);
	
	cic_axis ax0, ax1, ax2;
	const double *arr = args.gdata;
	
	arr = cic_axis_init(ax0, arr, x, args.gn0, args.gs0);
	arr = cic_axis_init(ax1, arr, y, args.gn1, args.gs1);
	arr = cic_axis_init(ax2, arr, z, args.gn2, args.gs2);

	rdata[i] = cic_interp_3d(ax0, ax1, ax2, arr);
    }

    return ret;
}

// -------------------------------------------------------------------------------------------------
//
// CIC gridding.


inline void cic_grid_1d(cic_axis &ax, double *arr, double val)
{
    arr[0] += ax.w0 * val;
    arr[ax.stride] += ax.w1 * val;
}

inline void cic_grid_2d(cic_axis &ax0, cic_axis &ax1, double *arr, double val)
{
    cic_grid_1d(ax1, arr, ax0.w0 * val);
    cic_grid_1d(ax1, arr + ax0.stride, ax0.w1 * val);
}

inline void cic_grid_3d(cic_axis &ax0, cic_axis &ax1, cic_axis &ax2, double *arr, double val)
{
    cic_grid_2d(ax1, ax2, arr, ax0.w0 * val);
    cic_grid_2d(ax1, ax2, arr + ax0.stride, ax0.w1 * val);
}


void cic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights, double lpos0, double lpos1, double lpos2, double pixsize)
{
    interpolation_args<double> args(grid, points, weights, lpos0, lpos1, lpos2, pixsize);

    for (long i = 0; i < args.npoints; i++) {
	double x, y, z, w;
	args.get_xyzw(i, x, y, z, w);
	
	cic_axis ax0, ax1, ax2;
	double *arr = args.gdata;
	
	arr = cic_axis_init(ax0, arr, x, args.gn0, args.gs0);
	arr = cic_axis_init(ax1, arr, y, args.gn1, args.gs1);
	arr = cic_axis_init(ax2, arr, z, args.gn2, args.gs2);

	cic_grid_3d(ax0, ax1, ax2, arr, w);
    }
}
