#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "elliptic_integral.h"

/* This program can be used to precalculate the binary files containing
 * quantities necessary for calculation of microlensing magnification
 * for extended sources. See also Bozza et al. 2018, MNRAS 479, 5157.
 * A(u,rho) = (u^2+2)/u/sqrt(u^2+4) * f0(rho/u,rho) for u > rho
 * A(u,rho) = sqrt(1+4/rho^2) * fi(u/rho,rho) for u < rho
 * The functions f0 and fi can be read from Witt & Mao (1994)
 * in terms of elliptic integrals. The function finite_source calculates
 * f0 and f1 as a function of z = rho/u (or u/rho) and rho.
 * 
 * Results are saved in binary files func0.dat and func1.dat. Both files
 * contain (Z_STEPS+1)*(LOGRHO_STEPS+1) double values, corresponding
 * to different values of z and logrho ("rows" corresponds to different
 * values of z, "columns" -- logrho). 
 * 
 * Compilation:
 * gcc -Wall -o precalculate_table precalculate_table.c elliptic_integral.c -lm
 * 
 * P. Mroz @ IPAC, 30 Apr 2019
 */

#define Z_MIN 0.0
#define Z_MAX 1.0
#define LOGRHO_MIN -3.0
#define LOGRHO_MAX 2.0
#define Z_STEPS 128
#define LOGRHO_STEPS 256

void finite_source (double *f0, double *fi, double z, double rho) {
    
    double n,k,t,E,F,Pi;
    double w1,w2,w3,norm1,norm2;
    
    n = 4.0*z/(z+1.0)/(z+1.0);
    k = sqrt(4.0*n/(4.0+rho*rho*(z-1.0)*(z-1.0)));
    
    t = 1.0/rho;
    
    E = elliptic_ek(k);
    F = elliptic_fk(k);
    Pi = elliptic_pik(n,k);
    
    w1 = 0.5*(z+1.0)*sqrt(4.0*t*t+(z-1.0)*(z-1.0));
    w2 = (z-1.0)*(4.0*t*t+0.5*(z*z-1.0))/sqrt(4.0*t*t+(z-1.0)*(z-1.0));
    w3 = 2.0*(z-1.0)*(z-1.0)*(1.0+t*t)/sqrt(4.0*t*t+(z-1.0)*(z-1.0))/(z+1.0);
    
    norm1 = (z*z+2.0*t*t)/z/sqrt(z*z+4.0*t*t);
    norm2 = sqrt(1.0+4.0*t*t);

    *f0 = (w1*E-w2*F+w3*Pi)/M_PI/norm1;
    *fi = (w1*E-w2*F+w3*Pi)/M_PI/norm2;
    
}

int main (int argc, char *argv[]) {

    double z_min,z_max,logrho_min,logrho_max,d_z,d_logrho;
    double z,logrho,rho,f0,f1,A;
    int z_steps,logrho_steps,i,j;

    FILE *file0,*file1;

    /* Setting up the range and spacing of the grid */
    z_min = Z_MIN;
    z_max = Z_MAX;
    logrho_min = LOGRHO_MIN;
    logrho_max = LOGRHO_MAX;
    z_steps = Z_STEPS;
    logrho_steps = LOGRHO_STEPS;
    
    d_z = (z_max-z_min)/z_steps;
    d_logrho = (logrho_max-logrho_min)/logrho_steps;
    
    file0 = fopen("func0.dat","wb");
    file1 = fopen("func1.dat","wb");
    
    /* If z == 0, both correction functions f1(z=0) = f0(z=0) = 1 */
    for (j=0; j<=logrho_steps; j++) {
        f1 = 1.0; f0 = 1.0;
        fwrite(&f1,sizeof(double),1,file1);
        fwrite(&f0,sizeof(double),1,file0);
    }
    /* If 0 < z < 1, f1(z) and f2(z) can be calculated using elliptic
     * integrals
     */
    for (i=1; i<z_steps; i++) {
        for (j=0; j<=logrho_steps; j++) {
            z = z_min+i*d_z;
            logrho = logrho_min+j*d_logrho;
            rho = pow(10.0,logrho);
            finite_source(&f0,&f1,z,rho);
            fwrite(&f1,sizeof f1,1,file1);
            finite_source(&f0,&f1,1.0/z,rho);
            fwrite(&f0,sizeof f0,1,file0);
        }
    }
    /* If z = 1, f1(z=1) and f2(z=2) can be calculated using analytic
     * expression that was derived by Witt & Mao (eq. 11)
     */
    z = 1.0;
    for (j=0; j<=logrho_steps; j++) {
        logrho = logrho_min+j*d_logrho;
        rho = pow(10.0,logrho);
        A = (0.5*M_PI+asin((rho*rho-1.0)/(rho*rho+1.0)))*(1.0+rho*rho)/rho/rho+2.0/rho;
        A /= M_PI;
        f1 = A/sqrt(1.0+4.0/rho/rho);
        fwrite(&f1,sizeof(double),1,file1);
        f0 = A*z*sqrt(z*z+4.0/rho/rho)/(z*z+2.0/rho/rho);
        fwrite(&f0,sizeof(double),1,file0);
    }
    
    fclose(file0);
    fclose(file1);

    return 0;
}
