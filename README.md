#Finite

Finite is package for calculating microlensing magnifications in the extended-source single-lens regime. The magnifications are evaluated using a pre-calculated grid as proposed by Bozza et al. (2018, MNRAS 479, 5157):
A(u,rho) = (u^2+2)/u/sqrt(u^2+4) * f0(rho/u,rho) for u > rho,
A(u,rho) = sqrt(1+4/rho^2) * fi(u/rho,rho) for u < rho.
The pre-calculated values of f0 and fi (as a function of z = rho/u or z = u/rho and rho) are read from binary files func0.dat and func1.dat. The magnifications are calculated using a linear interpolation between the grid points in z and logrho.

Files func0.dat and func1.dat should be pre-calculated using precalculate_table.
