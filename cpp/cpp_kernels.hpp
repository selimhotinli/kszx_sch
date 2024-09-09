#ifndef _KSZX_CPP_KERNELS_HPP
#define _KSZX_CPP_KERNELS_HPP

#include <climits>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


// -------------------------------------------------------------------------------------------------
//
// These functions are exported to python (see cpp_kernels.cpp for pybind11 boilerplate).


extern py::array_t<double> cic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points,
					      double lpos0, double lpos1, double lpos2, double pixsize);

extern py::array_t<double> cubic_interpolate_3d(py::array_t<const double> &grid, py::array_t<const double> &points,
						double lpos0, double lpos1, double lpos2, double pixsize);


extern void cic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights,
			double lpos0, double lpos1, double lpos2, double pixsize);

extern void cubic_grid_3d(py::array_t<double> &grid, py::array_t<const double> &points, py::array_t<const double> &weights,
			  double lpos0, double lpos1, double lpos2, double pixsize);


extern py::tuple estimate_power_spectrum(py::list map_list, py::array_t<const double> &k_delim,
					 py::array_t<const long> &npix, py::array_t<const double> &kf,
					 double box_volume);

extern py::tuple kbin_average(py::array_t<const double> &fk, py::array_t<const double> &k_delim,
			      py::array_t<const long> &npix, py::array_t<const double> &kf);


// -------------------------------------------------------------------------------------------------
//
// These inline functions are used internally.


// Hardware branch predictor hint.
#ifndef _unlikely
#define _unlikely(x)  (__builtin_expect(!!(x), 0))
#endif


template<typename T>
inline int get_int_shape(const py::array_t<T> &arr, int d)
{
    long s = arr.shape(d);

    if (_unlikely((s > INT_MAX)))
	throw std::runtime_error("kszx.cpp_kernels: array axis length >= 2^31?!");

    return int(s);
}


template<typename T>
inline int get_int_stride(const py::array_t<T> &arr, int d)
{
    if (_unlikely(arr.strides(d) % sizeof(T)))
	throw std::runtime_error("kszx.cpp_kernels: unaligned array stride?!");
    
    long s = arr.strides(d) / sizeof(T);

    if (_unlikely((s < INT_MIN) || (s > INT_MAX)))
	throw std::runtime_error("kszx.cpp_kernels: array stride >= 2^31?!");

    return int(s);
}

    
#endif  // _KSZX_CPP_KERNELS_HPP
