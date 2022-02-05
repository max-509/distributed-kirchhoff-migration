#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "implementation.h"
#include <stdexcept>

namespace py = pybind11;
// Фун-кия буффера и сохранения данных
template < typename T,typename T1>
py::array_t<double> calculate_migration(py::array_t<T> seismogram, py::array_t<T1> timeneiron, float dt) {
    py::buffer_info seismogram_info = seismogram.request();
    py::buffer_info timeneiron_info = timeneiron.request();
    // обработка сырых данных массива seismogram
    int n_traces = seismogram_info.shape[0];
    int n_samples = seismogram_info.shape[1];
    int n_node = timeneiron_info.shape[0];
    int n_point=timeneiron_info.shape[1];

    if (n_traces!=n_point)
        throw std::runtime_error("Input mismatch.Shape: data_trace(n_traces,n_samples),data_time(n_node,n_samples)");
    if (n_traces==n_point){
        T *ptr = (T *)seismogram_info.ptr;
        T1 *ptr1 = (T1 *)timeneiron_info.ptr;

        // обработка массива timeneiro

        auto sumAmp_node= py::array_t<double>(seismogram_info.size);
        py::buffer_info sumAmp_node_info = sumAmp_node.request();
        double *ptr2 = (double *)sumAmp_node_info.ptr;

        py::gil_scoped_release release;
        calculate_sumAmp_node(ptr,ptr1,ptr2,n_traces, n_samples, n_node,dt);
        py::gil_scoped_acquire acquire;
        return sumAmp_node;

    }

}
PYBIND11_MODULE(_migration, m) {
    m.def("calculate_migration", calculate_migration<double,float>);
    m.def("calculate_migration", calculate_migration<double,double>);
    m.def("calculate_migration", calculate_migration<float,double>);
    m.def("calculate_migration", calculate_migration<float,float>);
}