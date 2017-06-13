import numpy as np
import matplotlib.pyplot as plt
import emcee
from covariance_estimation.analytical_covolution import (log_posterior, posterior)
from scipy import integrate
import seaborn

ndim = 1
nwalkers = 10

zeta = 1.
sum_y2 = 21.9802082382
n = 20
omega_bar = sum_y2/float(n) - zeta

p_0 = [(1+np.random.rand(ndim)) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[zeta, sum_y2, n])
sampler.run_mcmc(p_0, 2000)
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
omega_min = 0.
omega_max = 5.

plt.hist(samples, histtype='step', normed=True, range=(omega_min, omega_max), bins=20)

nrm = integrate.quad(posterior, a=omega_min, b=omega_max, args=(zeta, sum_y2, n))
omega = np.linspace(omega_min, omega_max, 200)
post = np.zeros(len(omega))
for i in range(len(omega)):
    post[i] = posterior(omega[i], zeta, sum_y2, n)

plt.plot(omega, post/nrm[0])
plt.plot([omega_bar, omega_bar], [0., 1.6])
plt.ylim(0, 1.6)
plt.show()
