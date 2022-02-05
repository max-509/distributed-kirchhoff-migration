#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf
import configparser

def main():

    def createConfig(path):
    config=configparser.ConfigParser()
    config.add_section("Settings")
    config.add_section("Directories")
    config.set("Settings", "Координата начальной точки по X", input('Координата начальной точки по X:'))
    config.set("Settings", "Координата конечной точки по X", input('Координата конечной точки по X:'))
    config.set("Settings", "Координата начальной точки по Z", input('Координата начальной точки по Z:'))
    config.set("Settings", "Координата конечной точки по Z", input('Координата конечной точки по Z:'))
    config.set("Settings", "Количество точек среды по оси X", input('Количество точек среды по оси X:'))
    config.set("Settings", "Количество точек среды по оси Z", input('Количество точек среды по оси Z:'))
    config.set("Directories", "Данные о параметрах источника и приемника", "C:\Users\Руслан\Desktop\syst_obs.csv")
    config.set("Directories", "Сейсмограмма", "")
    with open(path, "w") as config_file:
        config.write(config_file)

    path = "settings.ini"
    createConfig(path)

    config=configparser.ConfigParser()
    config.read('settings.ini')
    directories = config['Directories']

    path_nn = "/content/gdrive/MyDrive/Colab Notebooks/EiconalNN.pb"
    loaded = tf.keras.models.load_model(path_nn)

    data_set = pd.read_csv(directories['Данные о параметрах источника и приемника'])
    seism_trace = np.load(directories['Сейсмограмма'])

    data_set_source = data_set.drop(['FFID', 'RECZ', 'RECX', 'SOUZ', 'CDPX_bin'], axis=1)
    data_set_receiver = data_set.drop(['FFID', 'SOUX', 'SOUZ', 'RECZ', 'CDPX_bin'], axis=1)

    parametres = config['Settings']

    nx = parametres["Количество точек среды по оси X"]
    nz = parametres["Количество точек среды по оси Z"]
    x0 = parametres['Координата начальной точки по X']
    x1 = parametres["Координата конечной точки по X"]
    z0 = parametres["Координата начальной точки по Z"]
    z1 = parametres["Координата конечной точки по Z"]
    dx = (x1-x0)/(nx-1)
    dz = (z1-z0)/(nz-1)
    data_set_source = data_set.drop(['FFID', 'RECZ', 'RECX', 'SOUZ', 'CDPX_bin'], axis=1)
    data_set_receiver = data_set.drop(['FFID', 'SOUX', 'SOUZ', 'RECZ', 'CDPX_bin'], axis=1)

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

    data = cartesian_product(masx, masz)

    df = pd.DataFrame(data, columns = ["RECX", 'RECZ'])
    df = df[:20000]

    SOUX=np.array(data_set_source['SOUX'])
    POINTX=np.array(df['RECX'])
    POINTZ=np.array(df['RECZ'])

    itog_source_point=pd.DataFrame([SOUX,POINTZ,POINTX]).T
    itog_source_point.columns=['SOUX','POINTZ','POINTX']

    RECX=np.array(data_set_receiver['RECX'])

    itog_point_receiver=pd.DataFrame([RECX,POINTZ,POINTX]).T
    itog_point_receiver.columns=['RECX','POINTZ','POINTX']

    if rank == 0:
        data_trace = seism_trace
        data1_S = itog_point_receiver
        data1_R = itog_source_point
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

if __name__ == '__main__':
    main()