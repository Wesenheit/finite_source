# Finite

Finite is package for calculating microlensing magnifications in the extended-source single-lens regime. 

The magnifications are evaluated using a pre-calculated grid as proposed by Bozza et al. (2018, MNRAS 479, 5157):

A(u,rho) = (u^2+2)/u/sqrt(u^2+4) * f0(rho/u,rho) for u > rho,

A(u,rho) = sqrt(1+4/rho^2) * fi(u/rho,rho) for u < rho.

The pre-calculated values of f0 and fi (as a function of z = rho/u or z = u/rho and rho) are read from binary files func0.dat and func1.dat. The magnifications are calculated using a linear interpolation between the grid points in z and logrho.

Files func0.dat and func1.dat should be pre-calculated using precalculate_table.

## How to install?

```
git clone https://github.com/przemekmroz/finite_source
./compile.do
```

## Usage

### Python

```
import Finite

# reading pre-computed tables

f0 = np.fromfile('func0.dat')
f1 = np.fromfile('func1.dat')


u = 3.0
rho = 5.0
Gamma = 0.5
Lambda = 0.0
n_rings = 128

A = Finite.ampl_ld(u,rho,Gamma,Lambda,f0,f1,n_rings)
```

### C

```
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "finite.h"

int main (int argc, char *argv[]) {
    double func0[(Z_STEPS+1)*(LOGRHO_STEPS+1)];
    double func1[(Z_STEPS+1)*(LOGRHO_STEPS+1)];
    int i,j,n_rings;
    double A,u,rho,Gamma,Lambda;

    u = 0.1;
    rho = 1.0;
    Gamma = 0.5;
    Lambda = 0.5;

    n_rings=128;
    i = read_binary_table(func0,"../func0.dat",Z_STEPS+1,LOGRHO_STEPS+1);
    j = read_binary_table(func1,"../func1.dat",Z_STEPS+1,LOGRHO_STEPS+1);
    A = ampl(u,rho,func0,func1);
    printf("%f\n",A);
    A = ampl_ld(u,rho,Gamma,Lambda,func0,func1,n_rings);
    printf("%f\n",A);
    return 0;
}

```
