#ifndef MIGRATION_ON_C_IMPLEMENTATION_H
#define MIGRATION_ON_C_IMPLEMENTATION_H
#include <vector>
void do_result(double *ptr, double *ptr1,double *ptr2, int n_traces, int n_samples,int n_node, float dt) {
    // Время трассы от 0 до конечного значения через dt

    int len_nx_tn=n_traces; // Создаю длину строки для t нейронки
    for (int i = 0; i < n_node; i++) { // Первый элемент строки t нейронки
        for (int j = 0; j < n_traces; j++){ // Оставшиеся элементы в строке
            int indx;
            indx = int(ptr1[i*len_nx_tn+j]/dt);
            ptr2[i]+=ptr[j*n_samples+indx];
            }
        }
    }


#endif //MIGRATION_ON_C_IMPLEMENTATION_H