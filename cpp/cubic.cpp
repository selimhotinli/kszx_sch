#include <iostream>
#include "cpp_kernels.hpp"
#include "interpolation_args.hpp"

using namespace std;


// FIXME lots of cut-and-paste between cic.cpp and cubic.cpp,
// that could be fixed with some C++ template magic.


struct cubic_axis
{
    int stride;
    double w0, w1, w2, w3;
};


// The template lets 'arr' be either (double *) or (const double *).
template<typename T>
inline T *cubic_axis_init(cubic_axis &ax, T *arr, double x, int n, int stride)
{
    int i = int(x);
    bool bad = (i < 1) || (i >= n-2);

    if (_unlikely(bad))
	throw runtime_error("kszx.cpp_kernels.cubic_interpolate_3d(): point is out of bounds");

    double f = x - (double)i;
    ax.stride = stride;
    ax.w0 = -(f)*(f-1)*(f-2) / 6.0;
    ax.w1 = (f+1)*(f-1)*(f-2) / 2.0;
    ax.w2 = -(f+1)*(f)*(f-2) / 2.0;		
    ax.w3 = (f+1)*(f)*(f-1) / 6.0;
    
    return arr + long(i-1) * long(stride);  // Note (i-1) here.
}


// -------------------------------------------------------------------------------------------------
//
// Cubic interpolation.


inline double cubic_interp_1d(cubic_axis &ax, const double *arr)
{
    int s = ax.stride;
    return (ax.w0 * arr[0]) + (ax.w1 * arr[s]) + (ax.w2 * arr[2*s]) + (ax.w3 * arr[3*s]);
}

inline double cubic_interp_2d(cubic_axis &ax0, cubic_axis &ax1, const double *arr)
{
    int s = ax0.stride;
    double f0 = cubic_interp_1d(ax1, arr);
    double f1 = cubic_interp_1d(ax1, arr+s);
    double f2 = cubic_interp_1d(ax1, arr+2*s);
    double f3 = cubic_interp_1d(ax1, arr+3*s);
    return (ax0.w0 * f0) + (ax0.w1 * f1) + (ax0.w2 * f2) + (ax0.w3 * f3);
}

inline double cubic_interp_3d(cubic_axis &ax0, cubic_axis &ax1, cubic_axis &ax2, const double *arr)
{
    int s = ax0.stride;
    double f0 = cubic_interp_2d(ax1, ax2, arr);
    double f1 = cubic_interp_2d(ax1, ax2, arr+s);
    double f2 = cubic_interp_2d(ax1, ax2, arr+2*s);
    double f3 = cubic_interp_2d(ax1, ax2, arr+3*s);
    return (ax0.w0 * f0) + (ax0.w1 * f1) + (ax0.w2 * f2) + (ax0.w3 * f3);
}


py::array_t<double> cubic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points, double lpos0, double lpos1, double lpos2, double pixsize)
{
    interpolation_args<const double> args(grid, points, lpos0, lpos1, lpos2, pixsize);

    py::array_t<double> ret({args.npoints});
    double *rdata = ret.mutable_data();

    for (long i = 0; i < args.npoints; i++) {
	double x, y, z;
	args.get_xyz(i, x, y, z);
	
	cubic_axis ax0, ax1, ax2;
	const double *arr = args.gdata;
	
	arr = cubic_axis_init(ax0, arr, x, args.gn0, args.gs0);
	arr = cubic_axis_init(ax1, arr, y, args.gn1, args.gs1);
	arr = cubic_axis_init(ax2, arr, z, args.gn2, args.gs2);

	rdata[i] = cubic_interp_3d(ax0, ax1, ax2, arr);
    }

    return ret;
}

// -------------------------------------------------------------------------------------------------
//
// Cubic gridding.


inline void cubic_grid_1d(cubic_axis &ax, double *arr, double val)
{
    int s = ax.stride;
    arr[0] += ax.w0 * val;
    arr[s] += ax.w1 * val;
    arr[2*s] += ax.w2 * val;
    arr[3*s] += ax.w3 * val;
}

inline void cubic_grid_2d(cubic_axis &ax0, cubic_axis &ax1, double *arr, double val)
{
    int s = ax0.stride;
    cubic_grid_1d(ax1, arr, ax0.w0 * val);
    cubic_grid_1d(ax1, arr+s, ax0.w1 * val);
    cubic_grid_1d(ax1, arr+2*s, ax0.w2 * val);
    cubic_grid_1d(ax1, arr+3*s, ax0.w3 * val);
}

inline void cubic_grid_3d(cubic_axis &ax0, cubic_axis &ax1, cubic_axis &ax2, double *arr, double val)
{
    int s = ax0.stride;
    cubic_grid_2d(ax1, ax2, arr, ax0.w0 * val);
    cubic_grid_2d(ax1, ax2, arr+s, ax0.w1 * val);
    cubic_grid_2d(ax1, ax2, arr+2*s, ax0.w2 * val);
    cubic_grid_2d(ax1, ax2, arr+3*s, ax0.w3 * val);
}


void cubic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights, double lpos0, double lpos1, double lpos2, double pixsize)
{
    interpolation_args<double> args(grid, points, weights, lpos0, lpos1, lpos2, pixsize);

    for (long i = 0; i < args.npoints; i++) {
	double x, y, z, w;
	args.get_xyzw(i, x, y, z, w);
	
	cubic_axis ax0, ax1, ax2;
	double *arr = args.gdata;
	
	arr = cubic_axis_init(ax0, arr, x, args.gn0, args.gs0);
	arr = cubic_axis_init(ax1, arr, y, args.gn1, args.gs1);
	arr = cubic_axis_init(ax2, arr, z, args.gn2, args.gs2);

	cubic_grid_3d(ax0, ax1, ax2, arr, w);
    }
}
