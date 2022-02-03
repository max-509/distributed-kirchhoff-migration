#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf

def main():



nx = 1000
nz = 50
x0 = 0
x1 = 4000
z0 = 0
z1 = 2000
dx = (x1-x0)/(nx-1)
dz = (z1-z0)/(nz-1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


path = "/content/gdrive/MyDrive/Colab Notebooks/EiconalNN.pb"
loaded = tf.keras.models.load_model(path)

data_set = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/syst_obs_csv.gz')
seism_trace = np.load('/content/gdrive/MyDrive/Colab Notebooks/seism.npy')

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
print ("Process = %s row = %s\n column = %s ----> neighbour_processes[UP] = %s neighbour_processes[DOWN] = %s neighbour_processes[LEFT] =%s neighbour_processes[RIGHT]=%s" %(rank, my_mpi_row, my_mpi_col,neighbour_processes[UP],neighbour_processes[DOWN], neighbour_processes[LEFT] , neighbour_processes[RIGHT]))

x_coor = dx*(nx-1)/grid_rows
z_coor = dz*(nz-1)/grid_column

#print(x_coor, z_coor)

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
#print(data)

df = pd.DataFrame(data, columns = ["RECX", 'RECZ'])
df = df[:20000]

SOUX=np.array(data_set_source['SOUX'])
CDPX=np.array(data_set_source['CDPX'])
#RECX=np.array(df['RECX'])
RECZ=np.array(df['RECZ'])

itog_source_point=pd.DataFrame([SOUX,RECZ,CDPX]).T
itog_source_point.columns=['SOUX','RECZ','CDPX']
#print(itog_source_point)

RECX_R=np.array(data_set_receiver['RECX'])
CDPX_R=np.array(data_set_receiver['CDPX'])

itog_point_receiver=pd.DataFrame([RECX_R,RECZ,CDPX_R]).T
itog_point_receiver.columns=['RECX_R','RECZ',"CDPX_R"]
#print(itog_point_receiver)
if rank == 0:
    data_trace = seism_trace
    data1_S = itog_point_receiver
    data1_R = itog_source_point
    data2 = data_trace
else:
    data1 = None
    data2 = None
data1_S = comm.bcast(data1_S, root=0)
data1_R = comm.bcast(data1_R, root=0)
data2 = comm.bcast(data2, root=0)

my_m1 = data1_S.shape[0]/size
my_matrix1 = data1_S[int(rank*my_m1):int((rank+1)*my_m1)]
my_m2 = data1_R.shape[0]/size
my_matrix2 = data1_R[int(rank*my_m1):int((rank+1)*my_m1)]
time1 = loaded.predict(my_matrix1)
time2 = loaded.predict(my_matrix2)

my_m2 = data2.shape[0]/size
my_matrix2 = data2[int(rank*my_m2):int((rank+1)*my_m2)]
print(str(time1), str(time2))

if __name__ == '__main__':
    main()