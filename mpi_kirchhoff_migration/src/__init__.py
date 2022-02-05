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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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

    if rank == 0:
        sources_coords = data_set_source['SOUX'].values.reshape(-1)
        receivers_coords = data_set_receiver['RECX'].values.reshape(-1)
        seismogramm = seism_trace
    else:
        sources_coords = None
        receivers_coords = None
        seismogramm = None
    sources_coords = comm.bcast(sources_coords, root=0)
    receivers_coords= comm.bcast(receivers_coords, root=0)
    seismogramm = comm.bcast(seismogramm, root=0)

    d_source = cartesian_product(sources_coords, masz, masx)
    d_receiver = cartesian_product(receivers_coords, masz, masx)
    my_m1 = d_source.shape[0]/size
    my_matrix1 = d_source[int(rank*my_m1):int((rank+1)*my_m1)]
    my_m2 = d_receiver.shape[0]/size
    my_matrix2 = d_receiver[int(rank*my_m2):int((rank+1)*my_m2)]
    my_m3 = seismogramm.shape[0]/size
    my_matrix3 = seismogramm[int(rank*my_m3):int((rank+1)*my_m3)]
    time1 = loaded.predict(itog_source_point)
    time2 = loaded.predict(itog_point_receiver)
    @numba.njit(parallel=True)
    def travel_times_sum(t1, t2): return t1+ t2
    travel_times_sum(time1, time2)
    print(travel_times_sum)
if __name__ == '__main__':
    main()