import numpy as np
from math import (log, exp)


def log_posterior(nu, zeta, y, x):
    omega = log(1. + exp(nu))
    if omega <= 0. or omega > 650.:
        return -np.inf
    n = len(x)
    assert n == 20
    assert zeta == 1
    return (-0.5*np.linalg.norm(y-x)**2)/zeta - 0.5*n*log(omega) \
        - (0.5*np.linalg.norm(x)**2)/omega + log(1. - exp(-omega))
