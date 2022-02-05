#ifndef MIGRATION_ON_C_IMPLEMENTATION_H
#define MIGRATION_ON_C_IMPLEMENTATION_H

#include <vector>
#include <iostream>
#include <omp.h>
#include <cstddef>
template < typename T,typename T1>
void calculate_sumAmp_node(T *ptr, T1 *ptr1, double *ptr2, int n_traces, int n_samples,
                           int n_node, float dt,std::ptrdiff_t strides_seismogram_info_y,
                           std::ptrdiff_t strides_seismogram_info_x,std::ptrdiff_t strides_timeneiron_info_y,
                           std::ptrdiff_t strides_timeneiron_info_x)
                           {

    #pragma omp parallel for
    for (int i = 0; i < n_node; i++) { // Первый элемент строки t нейронки
        for (int j = 0; j < n_traces; j++) { // Оставшиеся элементы в строке
            int indx;
            indx = int(ptr1[strides_timeneiron_info_y* i + j*strides_timeneiron_info_x] / dt);
            if (indx < n_samples) {
                ptr2[i] += ptr[j * strides_seismogram_info_y + indx*strides_seismogram_info_x];
            }
        }
    }
}

#endif //MIGRATION_ON_C_IMPLEMENTATION_H