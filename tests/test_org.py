def test_org():
    import numpy as np
    import Finite

    u0 = 3.0
    rho = 5.0
    Gamma = 0.5
    Lambda = 0.0
    times = np.linspace(-20.0, 20.0, 1000)
    n_rings = 128

    u = np.sqrt(u0 * u0 + times * times)

    A = np.array([Finite.ampl_ld(x, rho, Gamma, Lambda, n_rings) for x in u])
    A_org = np.load("tests/org.npy")
    assert np.mean(np.abs(A - A_org)) < 1e-6
