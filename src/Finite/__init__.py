import Finite_C
import numpy as np
from importlib.resources import files

func0 = np.fromfile(files("Finite").joinpath("func0.dat"))
func1 = np.fromfile(files("Finite").joinpath("func1.dat"))


def ampl(u, rho):
    return Finite_C.ampl_C(u, rho, func0, func1)


def ampl_ld(u, rho, Gamma, Lambda, n_rings):
    return Finite_C.ampl_ld_C(u, rho, Gamma, Lambda, func0, func1, n_rings)


def ampl_ld_array(u, rho, Gamma, Lambda, n_rings):
    return Finite_C.ampl_ld_array_C(u, rho, Gamma, Lambda, func0, func1, n_rings)
