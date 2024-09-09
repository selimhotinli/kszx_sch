#include "cpp_kernels.hpp"


PYBIND11_MODULE(cpp_kernels, m)
{
    // m.doc = "C++ kernels";

    m.def("cic_interpolate_3d", cic_interpolate_3d,
	  py::arg("grid"), py::arg("points"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"));

    m.def("cubic_interpolate_3d", cubic_interpolate_3d,
	  py::arg("grid"), py::arg("points"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"));

    m.def("cic_grid_3d", cic_grid_3d,
	  py::arg("grid"), py::arg("points"), py::arg("weights"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"));

    m.def("cubic_grid_3d", cubic_grid_3d,
	  py::arg("grid"), py::arg("points"), py::arg("weights"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"));

    m.def("estimate_power_spectrum", estimate_power_spectrum,
	  py::arg("map_list"), py::arg("k_delim"),
	  py::arg("npix"), py::arg("kf"),
	  py::arg("box_volume"));

    m.def("kbin_average", kbin_average,
	  py::arg("fk"), py::arg("k_delim"),
	  py::arg("npix"), py::arg("kf"));
}
