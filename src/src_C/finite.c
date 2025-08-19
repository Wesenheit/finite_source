#include "finite.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* This program can be used for calculation of microlensing
 * magnifications in the extended source single lens model. The
 * magnifications are evaluated using a pre-calculated grid as proposed
 * by Bozza et al. 2018, MNRAS 479, 5157:
 * A(u,rho) = (u^2+2)/u/sqrt(u^2+4) * f0(rho/u,rho) for u > rho
 * A(u,rho) = sqrt(1+4/rho^2) * fi(u/rho,rho) for u < rho
 * The pre-calculated values of f0 and fi (as a function of z = rho/u or
 * z = u/rho and rho) are read from binary files func0.dat and func1.dat
 * The magnifications are calculated using a linear interpolation
 * between the grid points in z and logrho.
 *
 * Files func0.dat and func1.dat should be pre-calculated using
 * precalculate_table.
 *
 * This file provides two functions:
 * double ampl (double u, double rho) which calculates the amplification
 * for a uniform source, and
 * double ampl_ld (double u, double rho, double Gamma, double Lambda)
 * which evaluates the amplification for a two-parameter limb darkening
 * profile of the source.
 * The latter function integrates the total magnification of concentric
 * annuli as described by Bozza et al. 2018.
 *
 * Proposed usage:
 * double func0[(Z_STEPS+1)*(LOGRHO_STEPS+1)];
 * double func1[(Z_STEPS+1)*(LOGRHO_STEPS+1)];
 * int i,j,n_rings;
 * double A,u,rho,Gamma,Lambda;
 *
 * n_rings=128;
 * i = read_binary_table(func0,"func0.dat",Z_STEPS+1,LOGRHO_STEPS+1);
 * j = read_binary_table(func1,"func1.dat",Z_STEPS+1,LOGRHO_STEPS+1);
 * A = ampl(u,rho,func0,func1);
 * A = ampl_ld(u,rho,Gamma,Lambda,func0,func1,n_rings);
 *
 * I also wrote a Python wrapper for these functions.
 *
 * P. Mroz @ IPAC, 30 Apr 2019
 */

int read_binary_table(double *tab, char *fname, int m, int n) {

  /* Reading binary file containing pre-calculated magnifications */

  FILE *fp;
  int i;

  fp = fopen(fname, "rb");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open file %s.\n", fname);
    return (1);
  }
  for (i = 0; i < m * n; i++) {
    fread(&tab[i], sizeof(double), 1, fp);
  }
  fclose(fp);
  return (0);
}

double interp_2d(double z, double logrho, double *tab) {

  /* Linear interpolation */

  int ip, jp;
  double res, fx, fy;

  if (z < Z_MIN)
    return 0.0;
  if (z >= Z_MAX)
    return 0.0;
  if (logrho < LOGRHO_MIN)
    return 0.0;
  if (logrho >= LOGRHO_MAX)
    return 0.0;
  ip = (int)((z - Z_MIN) * Z_STEPS / (Z_MAX - Z_MIN));
  jp = (int)((logrho - LOGRHO_MIN) * LOGRHO_STEPS / (LOGRHO_MAX - LOGRHO_MIN));
  /* Bilinear interpolation:
   * https://en.wikipedia.org/wiki/Bilinear_interpolation */
  fx = ip + 1 + (Z_MIN - z) * Z_STEPS / (Z_MAX - Z_MIN);
  fy =
      jp + 1 + (LOGRHO_MIN - logrho) * LOGRHO_STEPS / (LOGRHO_MAX - LOGRHO_MIN);
  res = tab[ip * (LOGRHO_STEPS + 1) + jp] * fx * fy;
  res += tab[(ip + 1) * (LOGRHO_STEPS + 1) + jp] * (1.0 - fx) * fy;
  res += tab[ip * (LOGRHO_STEPS + 1) + jp + 1] * fx * (1.0 - fy);
  res += tab[(ip + 1) * (LOGRHO_STEPS + 1) + jp + 1] * (1.0 - fx) * (1.0 - fy);
  return res;
}

double ampl(double u, double rho, double *func0, double *func1) {

  /* Amplification for a uniform source, see Bozza et al. 2018
   * and Witt & Mao 1994 for the equations. */

  double A;

  if (u > rho) {
    A = (u * u + 2.0) / u / sqrt(u * u + 4.0);
    /* Calculate finite source effects only if u < 10.0*rho */
    if (u <= 10.0 * rho)
      A *= interp_2d(rho / u, log10(rho), func0);
  } else if (u < rho) {
    A = sqrt(1.0 + 4.0 / rho / rho) * interp_2d(u / rho, log10(rho), func1);
  } else {
    A = (1.0 + rho * rho) *
        (0.5 * M_PI + asin((rho * rho - 1.0) / (rho * rho + 1.0))) / rho / rho;
    A += 2.0 / rho;
    A /= M_PI;
  }
  return A;
}

double cumulative_profile(double r, double Gamma, double Lambda) {

  /* Returns the cumulative profile at fractional radius r
   * as a function of Gamma and Lambda. See Eq. (16) in Bozza et al.
   */

  double F;
  F = (1.0 - Gamma - Lambda) * r * r;
  F += Gamma + Lambda - Gamma * pow(1.0 - r * r, 1.5);
  F -= Lambda * pow(1.0 - r * r, 1.25);
  return F;
}

double ampl_ld(double u, double rho, double Gamma, double Lambda, double *func0,
               double *func1, int nsteps) {

  /* Amplification for a limb-darkened source. The amplification is
   * calculated by integrating nsteps concentric annuli, using the
   * method that is outlined in Section 2 of Bozza et al. 2018 */

  if (u >= 10.0 * rho)
    return ampl(u, rho, func0, func1);
  if (Gamma == 0.0 && Lambda == 0.0)
    return ampl(u, rho, func0, func1);

  int i;
  double rad[nsteps + 1];
  double cum[nsteps + 1];
  double mu[nsteps + 1];
  double A, fi, wi;

  rad[0] = 0.0;
  mu[0] = 0.0;
  cum[0] = 0.0;
  for (i = 1; i <= nsteps; i++) {
    rad[i] = (double)i / (double)nsteps;
    mu[i] = ampl(u, rad[i] * rho, func0, func1);
    cum[i] = cumulative_profile(rad[i], Gamma, Lambda);
  }
  A = 0.0;
  for (i = 1; i <= nsteps; i++) {
    wi = mu[i] * rad[i] * rad[i] - mu[i - 1] * rad[i - 1] * rad[i - 1];
    fi = (cum[i] - cum[i - 1]) / (rad[i] * rad[i] - rad[i - 1] * rad[i - 1]);
    A += fi * wi;
  }

  return (A);
}
