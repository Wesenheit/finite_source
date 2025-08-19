# Finite

Finite is package for calculating microlensing magnifications in the extended-source single-lens regime. 

The magnifications are evaluated using a pre-calculated grid as proposed by Bozza et al. (2018, MNRAS 479, 5157):

A(u,rho) = (u^2+2)/u/sqrt(u^2+4) * f0(rho/u,rho) for u > rho,

A(u,rho) = sqrt(1+4/rho^2) * fi(u/rho,rho) for u < rho.

The pre-calculated values of f0 and fi (as a function of z = rho/u or z = u/rho and rho) are read from binary files func0.dat and func1.dat. The magnifications are calculated using a linear interpolation between the grid points in z and logrho.

Files func0.dat and func1.dat should be pre-calculated using precalculate_table at build time.

This particular version is designed to be installed directly with pip or other modern tools.


## How to install?

Just try using pip
```
pip install git+https://github.com/Wesenheit/finite_source.git
```
or with any other modern python tool like uv. 
## Usage

### Python

```
import Finite

u = 3.0
rho = 5.0
Gamma = 0.5
Lambda = 0.0
n_rings = 64

A = Finite.ampl_ld(u,rho,Gamma,Lambda,n_rings)

u = np.linspace(0.0,1.0,100)
A = Finite.ampl_ld_array(u,rho,Gamma,Lambda,n_rings)
```
