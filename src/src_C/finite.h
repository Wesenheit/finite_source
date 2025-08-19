#ifndef FINITE_H_
#define FINITE_H_

#define Z_MIN 0.0
#define Z_MAX 1.0
#define LOGRHO_MIN -3.0
#define LOGRHO_MAX 2.0
#define Z_STEPS 128
#define LOGRHO_STEPS 256

double ampl(double, double, double *, double *);
double ampl_ld(double, double, double, double, double *, double *, int);

#endif
