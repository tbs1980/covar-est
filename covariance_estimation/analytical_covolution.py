from math import (log, exp, sqrt)
import numpy as np


def log_posterior(omega, zeta, sum_y2, n):
    if omega <= 0.:
        return -np.inf
    return -0.5*n*log(zeta + omega) - 0.5*sum_y2/(zeta + omega)


def posterior(omega, zeta, sum_y2, n):
    if omega <= 0.:
        return 0.
    return exp(log_posterior(omega, zeta, sum_y2, n))


def log_posterior_jeffreys_prior(omega, zeta, sum_y2, n):
    if omega <= 0.:
        return -np.inf
    return log_posterior(omega, zeta, sum_y2, n) - log(omega)


def posterior_jeffreys_prior(omega, zeta, sum_y2, n):
    if omega <= 0.:
        return np.inf
    return posterior(omega, zeta, sum_y2, n)/omega


def log_posterior_hierarchical_jeffreys_prior(omega, zeta, sum_y2, n):
    if omega <= 0.:
        return -np.inf
    return log_posterior(omega, zeta, sum_y2, n) - 0.5*log(omega + zeta)


def posterior_hierarchical_jeffreys_prior(omega, zeta, sum_y2, n):
    if omega <= 0.:
        return np.inf
    return posterior(omega, zeta, sum_y2, n)/sqrt(omega + zeta)
