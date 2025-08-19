import matplotlib.pyplot as plt
import numpy as np
import time

import Finite

# reading pre-computed tables

f0 = np.fromfile('func0.dat')
f1 = np.fromfile('func1.dat')


u0 = 3.0
rho = 5.0
Gamma = 0.5
Lambda = 0.0
times = np.linspace(-20.0,20.0,1000)
n_rings = 128

u = np.sqrt(u0*u0+times*times)

start = time.time()
A = np.array([Finite.ampl_ld(x,rho,Gamma,Lambda,f0,f1,n_rings) for x in u])
end = time.time()

print ('Execution time: %.2f s'%(end-start))

plt.plot(times,2.5*np.log10(A))
plt.show()
