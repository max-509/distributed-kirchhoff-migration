import _migration
import numpy as np

seism = np.load(r"C:\Users\GLEB2001\Desktop\seism.npy")
trace=seism[:10]
time=[[] for i in range(5)]
for i in range(5):
    time[i].append((np.random.sample()))
    time[i].append((np.random.sample()))
timeneiron=np.array(time)
#(10, 1501)
res=_migration.array_buffer(seism,timeneiron)
print(res)
