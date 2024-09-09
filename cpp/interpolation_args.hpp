#ifndef _KSZX_INTERPOLATION_ARGS_HPP
#define _KSZX_INTERPOLATION_ARGS_HPP

#include "cpp_kernels.hpp"


// -------------------------------------------------------------------------------------------------
//
// interpolation_args: this helper class handles the arguments (grid, points, lpos, pixsize)
// that are common to all interpolation/gridding kernels. Used in cic.cpp, cubic.cpp, etc.
//
// FIXME currently assumes ndim=3.


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
	    throw std::runtime_error("expected 'grid' to be a 3-d array");
	if (points.ndim() != 2)
	    throw std::runtime_error("expected 'points' to be a 2-d array");
	if (points.shape(1) != 3)
	    throw std::runtime_error("expected 'points' to be a shape (N,3) array");
	if (pixsize <= 0)
	    throw std::runtime_error("expected pixsize > 0");

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
	    throw std::runtime_error("expected all grid dimensions >= 2");
	
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
	    throw std::runtime_error("expected 'weights' to be a 1-d array");
	if (weights.shape(0) != npoints)
	    throw std::runtime_error("expected 'weights' to be a length-N array, where N=points.shape[0]");

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


#endif  // _KSZX_INTERPOLATION_ARGS_HPP
