#!/usr/bin/env python
import os.path

from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf
import configparser
import numba
import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor

from _migration import calculate_migration


@numba.njit(parallel=True)
def travel_times_sum(t1, t2):
    return t1 + t2


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def main():
    if len(sys.argv) != 2:
        raise AttributeError("Error: need pass path to setting.ini file")

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    directories = config['Directories']


    data_set = pd.read_csv(directories['data_source-receiver'], delimiter=',')
    seism_trace = np.load(directories['seismogramma'])
    path_to_result = directories['path_to_result']
    data_set_source = data_set['SOUX']
    data_set_receiver = data_set['RECX']

    parametres = config['Settings']

    nx = int(parametres["number_of_x_points"])
    nz = int(parametres["number_of_z_points"])
    x0 = float(parametres['starting_z_coord'])
    x1 = float(parametres["ending_z_coord"])
    z0 = float(parametres["starting_z_coord"])
    z1 = float(parametres["ending_z_coord"])
    dt = float(parametres["dt"])
    dx = (x1 - x0) / (nx - 1)
    dz = (z1 - z0) / (nz - 1)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[rank], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

    loaded = tf.keras.models.load_model(directories['neural_network'])

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    neighbour_processes = [0, 0, 0, 0]
    for grid_cols in range(int(np.floor(np.sqrt(size))), size + 1):
        if size % grid_cols == 0:
            grid_rows = size // grid_cols
            break

    cartesian_communicator = comm.Create_cart((grid_rows, grid_cols), periods=(False, False), reorder=True)
    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)
    neighbour_processes[UP], neighbour_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbour_processes[LEFT], neighbour_processes[RIGHT] = cartesian_communicator.Shift(1, 1)

    n = nx // grid_cols
    m = nz // grid_rows
    x_points = n
    z_points = m

    if my_mpi_col == grid_cols - 1:
        x_points += (nx % grid_cols)
    if my_mpi_row == grid_rows - 1:
        z_points += (nz % grid_rows)

    x_begin = my_mpi_col * (n * dx)
    x_end = x_begin + (x_points * dx)
    z_begin = my_mpi_row * (m * dz)
    z_end = z_begin + (z_points * dz)

    masx = np.linspace(x_begin, x_end, x_points)
    masz = np.linspace(z_begin, z_end, z_points)

    if rank == 0:
        sources_coords = data_set_source.values.reshape(-1)
        receivers_coords = data_set_receiver.values.reshape(-1)
        seismogramm = seism_trace
    else:
        sources_coords = None
        receivers_coords = None
        seismogramm = None
    sources_coords = comm.bcast(sources_coords, root=0)
    receivers_coords = comm.bcast(receivers_coords, root=0)
    seismogramm = comm.bcast(seismogramm, root=0)

    d_source = cartesian_product(sources_coords, masz, masx)
    print(f'D_SOURCE_SHAPE: {d_source.shape}')
    d_receiver = cartesian_product(receivers_coords, masz, masx)
    print(f'D_RECEIVER_SHAPE: {d_receiver.shape}')

    def times_calculator(input_coords):
        return loaded.predict(input_coords, batch_size=(2 ** 15))

    n_processes = 1
    with ThreadPoolExecutor(n_processes) as executor:
        def split(d):
            n_sources_points_pairs = d.shape[0]
            part = n_sources_points_pairs // n_processes
            split_parts = []
            for i in range(n_processes - 1):
                split_parts.append((i + 1) * part)
            split_parts.append(n_processes * part + (n_sources_points_pairs % n_processes))
            source_travel_times_splits = np.concatenate(list(executor.map(times_calculator, np.split(d, split_parts)[:-1])))
            return source_travel_times_splits

        # time1 = split(d_source)
        # time2 = split(d_receiver)
        time1 = times_calculator(d_source)
        time2 = times_calculator(d_receiver)

    travel_times = travel_times_sum(time1, time2).reshape(-1, len(masz) * len(masx)).T

    result = calculate_migration(seismogramm, travel_times, dt).reshape(m, n)

    np.save(os.path.join(path_to_result, f'result{my_mpi_row}{my_mpi_col}'), result)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    '''
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
            print(e)
    '''
    # tf.debugging.set_log_device_placement(True)
    # physical_devices = tf.config.list_physical_devices('GPU')
    # for device in physical_devices:
    #     tf.config.experimental.set_memory_growth(device, True)
    # with tf.device('/job:localhost/replica:0/task:0/device:GPU:1'):
    main()
