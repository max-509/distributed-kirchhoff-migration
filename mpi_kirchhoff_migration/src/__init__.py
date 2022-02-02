#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size=comm.size

    seism_trace = np.load(r'C:\Users\Руслан\Downloads\seism.npy') #Загружаю данные со сейсмограммы
    seism_trace

    data_set = pd.read_csv(r'C:\Users\Руслан\Desktop\syst_obs.csv') #Загружаю данные с поверхностными объектами
    data_set = data_set.drop(['FFID', 'RECZ', 'CDPX', 'CDPX_bin'], axis=1) # Избавляюсь от лишних данных
    dat_set = data_set.to_numpy() #Преобразую в формат массива

    if rank == 0:
        data_source = dat_set
        data_trace = seism_trace
        data1 = data_source
        data2 = data_trace
    else:
        data1 = None
        data2 = None
    data1 = comm.bcast(data1, root=0)
    data2 = comm.bcast(data2, root=0)

    my_m1 = data1.shape[0]/size
    my_matrix1 = data1[int(rank*my_m1):int((rank+1)*my_m1)] #узел с данными для обработки нейронкой

    my_m2 = data2.shape[0]/size
    my_matrix2 = data2[int(rank*my_m2):int((rank+1)*my_m2)] #данные для Глеба
if __name__ == '__main__':
    main()