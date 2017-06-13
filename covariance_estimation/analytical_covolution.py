from math import log, exp
import numpy as np


def log_posterior(omega, zeta, sum_y2, n):
    if omega < 0.:
        return -np.inf
    return -0.5*n*log(zeta + omega) - 0.5*sum_y2/(zeta + omega)


def posterior(omega, zeta, sum_y2, n):
    if omega < 0.:
        return 0.
    return exp(log_posterior(omega, zeta, sum_y2, n))