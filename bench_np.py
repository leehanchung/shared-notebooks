import time
import numpy as np


np.random.seed(0)
#np.__config__.info()
np.show_config()

n = 20000

A = np.random.randn(n, n).astype('float64')
B = np.random.randn(n, n).astype('float64')
start_time = time.time()
nrm = np.linalg.norm(A @ B)
print(" took {} seconds".format(time.time() - start_time))
print("norm = ",nrm)
