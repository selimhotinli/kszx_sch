#include "cpp_kernels.hpp"
#include "interpolation_args.hpp"

using namespace std;


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
