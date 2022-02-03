#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "implementation.h"

namespace py = pybind11;

// Фун-кия буффера и сохранения данных

py::array_t<double> array_buffer(py::array_t<double> seismogram, py::array_t<double> timeneiron, int dt) {
    py::buffer_info seismogram_info = seismogram.request();
    py::buffer_info timeneiron_info = timeneiron.request();

    // обработка сырых данных массива seismogram
    int n_traces = seismogram_info.shape[0];
    int n_samples = seismogram_info.shape[1];
    int n_node = timeneiron_info.shape[0];
    int n_points=timeneiron_info.shape[1];

    double *ptr = (double *)seismogram_info.ptr;
    double *ptr1 = (double *)timeneiron_info.ptr;

    // обработка массива timeneiro

    auto result = py::array_t<double>(seismogram_info.size);
    py::buffer_info result_info = result.request();
    double *ptr2 = (double *)result_info.ptr;

    do_result(ptr,ptr1,ptr2,n_traces, n_samples, n_node,dt);

    return result;
}

PYBIND11_MODULE(_migration, m) {
    m.def("array_buffer", &array_buffer);
}