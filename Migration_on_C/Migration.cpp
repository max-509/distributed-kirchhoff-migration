#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "implementation.h"

namespace py = pybind11;

// Фун-кия буффера и сохранения данных

py::array_t<double> array_buffer(py::array_t<double> seismogram, py::array_t<double> timeneiron) {
    py::buffer_info seismogram_info = seismogram.request();
    py::buffer_info timeneiron_info = timeneiron.request();

    auto size_seismogram = py::array_t<double>(seismogram_info.size);
    auto n_traces=size_seismogram[0];
    auto n_samples=n_traces[1];

    double *ptr = (double *)seismogram_info.ptr;
    double *ptr1 = (double *)timeneiron_info.ptr;


    result(ptr, ptr1, n_traces, n_samples,n_points) ;

//    return result;
}

PYBIND11_MODULE(_migration, m) {
    m.def("array_buffer", &array_buffer);
}