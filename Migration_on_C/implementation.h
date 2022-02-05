#ifndef MIGRATION_ON_C_IMPLEMENTATION_H
#define MIGRATION_ON_C_IMPLEMENTATION_H

#include <vector>
#include <iostream>
#include <omp.h>
template < typename T,typename T1>
void calculate_sumAmp_node(T *ptr, T1 *ptr1, double *ptr2, int n_traces, int n_samples, int n_node, float dt) {
    // Время трассы от 0 до конечного значения через dt

    int len_nx_tn = n_traces; // Создаю длину строки для t нейронки
    #pragma omp parallel for
    for (int i = 0; i < n_node; i++) { // Первый элемент строки t нейронки
        for (int j = 0; j < n_traces; j++) { // Оставшиеся элементы в строке
            int indx;
            indx = int(ptr1[i * len_nx_tn + j] / dt);
            if (indx < n_samples) {
                ptr2[i] += ptr[j * n_samples + indx];
            }
        }
    }
}

#endif //MIGRATION_ON_C_IMPLEMENTATION_H