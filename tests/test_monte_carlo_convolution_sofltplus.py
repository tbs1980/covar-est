import numpy as np
from covariance_estimation.monte_carlo_convolution_softplus import log_posterior
import emcee
import matplotlib.pyplot as plt
import seaborn
from covariance_estimation.analytical_covolution import posterior as ana_post
from math import (sqrt, log, exp)
from scipy import integrate


np.random.seed(31415)


def log_prob(x, zeta_in, y):
    return log_posterior(nu=x[-1], zeta=zeta_in, y=y, x=x[:-1])


omega_true = 1.
zeta = 1.
n = 20

y = np.random.normal(loc=0., scale=sqrt((omega_true + zeta)), size=n)


sum_y2 = np.linalg.norm(y)**2
omega_bar = sum_y2/float(n) - zeta

print("sum(y^2) = ", sum_y2)
print("omega-bar = ", omega_bar)


ndim = n+1
nwalkers = 60
p_0 = [np.random.rand(ndim) for i in range(nwalkers)]
for p in p_0:
    p[-1] = log(exp(omega_true + np.random.uniform(0, 1)) - 1.)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[zeta, y])
sampler.run_mcmc(p_0, 1000)
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

plt.hist(np.log(1. + np.exp(samples[:, 20])), histtype='step', normed=True, bins=20)

ax = plt.axis()
omega_min = max(ax[0], 0.)
omega_max = ax[1]

nrm = integrate.quad(ana_post, a=omega_min, b=omega_max, args=(zeta, sum_y2, n))
omega = np.linspace(omega_min, omega_max, 200)
post = np.zeros(len(omega))
for i in range(len(omega)):
    post[i] = ana_post(omega[i], zeta, sum_y2, n)

plt.plot(omega, post/nrm[0])
ax = plt.axis()
plt.plot([omega_bar, omega_bar], [ax[2], ax[3]])
plt.ylim(ax[2], ax[3])

plt.show()
