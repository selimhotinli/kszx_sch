#include "cpp_kernels.hpp"


PYBIND11_MODULE(cpp_kernels, m) {
    // m.doc = "C++ kernels";

    m.def("cic_interpolate_3d", cic_interpolate_3d,
	  py::arg("grid"), py::arg("points"),
	  py::arg("lpos0"), py::arg("lpos1"), py::arg("lpos2"),
	  py::arg("pixsize"));
}
