#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf
import configparser

def main():

    config=configparser.ConfigParser()
    config.read('settings.ini')
    directories = config['Directories']
    directories['data_source-receiver']

    loaded = tf.keras.models.load_model(directories['neural_network'])

    data_set = pd.read_csv(directories['data_source-receiver'])
    seism_trace = np.load(directories['seismogramma'])

    data_set_source = data_set.drop(['FFID', 'RECZ', 'RECX', 'SOUZ', 'CDPX_bin'], axis=1)
    data_set_receiver = data_set.drop(['FFID', 'SOUX', 'SOUZ', 'RECZ', 'CDPX_bin'], axis=1)

    parametres = config['Settings']

    nx = int(parametres["number_of_x_points"])
    nz = int(parametres["number_of_z_points"])
    x0 = int(parametres['starting_z_coord'])
    x1 = int(parametres["ending_z_coord"])
    z0 = int(parametres["starting_z_coord"])
    z1 = int(parametres["ending_z_coord"])
    dx = (x1-x0)/(nx-1)
    dz = (z1-z0)/(nz-1)

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    neighbour_processes = [0,0,0,0]

    grid_rows = int(np.floor(np.sqrt(size)))
    grid_column = size // grid_rows

    if grid_rows*grid_column > size:
        grid_column -= 1
    if grid_rows*grid_column > size:
        grid_rows -= 1

    cartesian_communicator = comm.Create_cart((grid_rows, grid_column),periods=(False, False),reorder=True)
    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)
    neighbour_processes[UP], neighbour_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbour_processes[LEFT], neighbour_processes[RIGHT]  = cartesian_communicator.Shift(1, 1)

    x_coor = dx*(nx-1)/grid_rows
    z_coor = dz*(nz-1)/grid_column

    n = nx//grid_rows
    m = nz//grid_column

    if my_mpi_row == grid_rows-1:
        n = n + nx%grid_rows
    if my_mpi_col == grid_column-1:
        m = m + nz%grid_column

    masx= np.linspace(my_mpi_row*x_coor, (my_mpi_row+1)*x_coor, n)
    masz = np.linspace(my_mpi_col*z_coor, (my_mpi_col+1)*z_coor, m)

    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)

    d_source = cartesian_product(data_set_source['SOUX'].values.reshape(-1), masz, masx)
    d_receiver = cartesian_product(data_set_receiver['RECX'].values.reshape(-1), masz, masx)

if rank == 0:
    data_trace = seism_trace
    data1_S = d_source
    data1_R = d_receiver
    data2 = data_trace
    else:
        data1_S = None
        data1_R = None
        data2 = None
    data1_S = comm.bcast(data1_S, root=0)
    data1_R = comm.bcast(data1_R, root=0)
    data2 = comm.bcast(data2, root=0)

    my_m1 = data1_S.shape[0]/size
    my_matrix1 = data1_S[int(rank*my_m1):int((rank+1)*my_m1)]
    my_m2 = data1_R.shape[0]/size
    my_matrix2 = data1_R[int(rank*my_m2):int((rank+1)*my_m2)]
    my_m3 = data2.shape[0]/size
    my_matrix3 = data2[int(rank*my_m3):int((rank+1)*my_m3)]
    time1 = loaded.predict(itog_source_point)
    time2 = loaded.predict(itog_point_receiver)
    time = time1 +time2
    print(str(time1), str(time2), time1.shape, time2.shape)
if __name__ == '__main__':
    main()