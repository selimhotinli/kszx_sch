#include "cpp_kernels.hpp"

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


// -------------------------------------------------------------------------------------------------


py::array_t<double> cic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points, double lpos0, double lpos1, double lpos2, double pixsize)
{
    // Argument checking.
    
    if (grid.ndim() != 3)
	throw runtime_error("kszx.cpp_kernels.cic_interpolate_3d(): expected 'grid' to be a 3-d array");
    if (points.ndim() != 2)
	throw runtime_error("kszx.cpp_kernels.cic_interpolate_3d(): expected 'points' to be a 2-d array");
    if (points.shape(1) != 3)
	throw runtime_error("kszx.cpp_kernels.cic_interpolate_3d(): expected 'points' to be a shape (N,3) array");
    if (pixsize <= 0)
	throw runtime_error("kszx.cpp_kernels.cic_interpolate_3d(): expected pixsize > 0");

    // Unpack python arrays into C++ pointers/ints.
    
    const double *gdata = grid.data();
    int gn0 = get_int_shape(grid, 0);
    int gn1 = get_int_shape(grid, 1);
    int gn2 = get_int_shape(grid, 2);
    int gs0 = get_int_stride(grid, 0);
    int gs1 = get_int_stride(grid, 1);
    int gs2 = get_int_stride(grid, 2);

    if ((gn0 < 2) || (gn1 < 2) || (gn2 < 2))
	throw runtime_error("kszx.cpp_kernels.cic_interpolate_3d(): expected all grid dimensions >= 2");
    
    const double *pdata = points.data();
    long npoints = points.shape(0);
    int ps0 = get_int_stride(points, 0);
    int ps1 = get_int_stride(points, 1);
    
    double rec_ps = 1.0 / pixsize;

    // Allocate returned array.
    
    py::array_t<double> ret({npoints});
    double *rdata = ret.mutable_data();

    // Main interpolation loop.
    
    for (long i = 0; i < npoints; i++) {
	// (x,y,z) = (points[i,:] - lpos) / pixsize
	double x = rec_ps * (pdata[i*ps0] - lpos0);
	double y = rec_ps * (pdata[i*ps0 + ps1] - lpos1);
	double z = rec_ps * (pdata[i*ps0 + 2*ps1] - lpos2);
	
	cic_axis ax0, ax1, ax2;
	const double *arr = gdata;
	
	arr = cic_axis_init(ax0, arr, x, gn0, gs0);
	arr = cic_axis_init(ax1, arr, y, gn1, gs1);
	arr = cic_axis_init(ax2, arr, z, gn2, gs2);

	rdata[i] = cic_interp_3d(ax0, ax1, ax2, arr);
    }

    return ret;
}
