import _migration
import numpy as np
import pandas as pd
import eikonalfm

seism = np.load(r"C:\Users\GLEB2001\Desktop\seism.npy")
data_set = pd.read_csv(r"C:\Users\GLEB2001\Desktop\syst_obs.csv")

dt = 0.001
dx = 10
dz = 10
x_set = np.arange(0,3970 + dx, dx)
z_set = np.arange(0,2000 + dz, dz)
velocity = np.zeros((len(z_set),len(x_set)))
velocity[:int(500/dz), :] = 2000
velocity[int(500/dz):int(1000 / dz), :] = 3000
velocity[int(1000/dz):int(1500 / dz), :] = 4000
velocity[int(1500/dz):int(2000 +dz / dz), :] = 5000

sources_x = data_set['SOUX'].drop_duplicates().values # 100 источников
inputu = np.zeros((20000, 3))
observed_time = np.zeros((20000, 1))
s = 0
for idx, sou in enumerate(sources_x):
    x_s = (0, int(sou/dx))
    ddx = (1.0, 1.0)
    tau_fm = eikonalfm.fast_marching(velocity, x_s, ddx, 2)
    for k in range(idx*200, (idx+1)*200):
        inputu[s] = np.array((sou,0, data_set['RECX'].values[k]))
        observed_time[s] = tau_fm[0,int(data_set['RECX'].values[k]/dx)]
        s += 1

#(10, 1501)
observed_time = observed_time.reshape(1,20000)
dt=0.001
res=_migration.calculate_migration(seism[::6, :],observed_time[:, ::6], dt)
print(res.shape)