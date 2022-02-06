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

    loaded = tf.keras.models.load_model(directories['neural_network'])
    data_set = pd.read_csv(directories['data_source-receiver'])
    seism_trace = np.load(directories['seismogramma'])
    path_to_result = directories['path_to_result']
    data_set_source = data_set["SOUX"]
    data_set_receiver = data_set["RECX"]

    parametres = config['Settings']

    nx = int(parametres["number_of_x_points"])
    nz = int(parametres["number_of_z_points"])
    x0 = int(parametres['starting_z_coord'])
    x1 = int(parametres["ending_z_coord"])
    z0 = int(parametres["starting_z_coord"])
    z1 = int(parametres["ending_z_coord"])
    dt = float(parametres["dt"])
    dx = (x1 - x0) / (nx - 1)
    dz = (z1 - z0) / (nz - 1)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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

    x_coor = dx * (nx - 1) / grid_cols
    z_coor = dz * (nz - 1) / grid_rows

    n = nx // grid_cols
    m = nz // grid_rows

    if my_mpi_col == grid_cols - 1:
        n = n + nx % grid_cols
    if my_mpi_row == grid_rows - 1:
        m = m + nz % grid_rows

    masx = np.linspace(my_mpi_col * x_coor, (my_mpi_col + 1) * x_coor, n)
    masz = np.linspace(my_mpi_row * z_coor, (my_mpi_row + 1) * z_coor, m)

    if rank == 0:
        sources_coords = data_set_source['SOUX'].values.reshape(-1)
        receivers_coords = data_set_receiver['RECX'].values.reshape(-1)
        seismogramm = seism_trace
    else:
        sources_coords = None
        receivers_coords = None
        seismogramm = None
    sources_coords = comm.bcast(sources_coords, root=0)
    receivers_coords = comm.bcast(receivers_coords, root=0)
    seismogramm = comm.bcast(seismogramm, root=0)

    d_source = cartesian_product(sources_coords, masz, masx)
    d_receiver = cartesian_product(receivers_coords, masz, masx)

    def times_calculator(input_coords):
        return loaded.predict(input_coords)

    n_processes = mp.cpu_count()
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

        time1 = split(d_source)
        time2 = split(d_receiver)

    travel_times = travel_times_sum(time1, time2).reshape(-1, len(masz) * len(masx)).T

    result = calculate_migration(seismogramm, travel_times, dt).reshape(m, n)

    np.save(os.path.join(path_to_result, f'result{my_mpi_row}{my_mpi_col}'), result)


if __name__ == '__main__':
    main()
