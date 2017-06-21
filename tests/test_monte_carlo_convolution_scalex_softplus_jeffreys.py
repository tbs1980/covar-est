from covariance_estimation.monte_carlo_convolution_scalex_softplus_jeffreys import log_posterior
from math import (sqrt, log, exp)
import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy import integrate
# from covariance_estimation.analytical_covolution import posterior_jeffreys_prior as ana_post
from covariance_estimation.analytical_covolution import posterior as ana_post


np.random.seed(31415)


def log_prob(s, zeta_in, y):
    return log_posterior(nu=s[-1], zeta=zeta_in, y=y, s=s[:-1])


omega_true = 5.
zeta = 1.
n = 20

y = np.random.normal(loc=0., scale=sqrt((omega_true + zeta)), size=n)

sum_y2 = np.linalg.norm(y)**2
print("sum(y^2) = ", sum_y2)

ndim = n+1
nwalkers = 200
p_0 = [np.random.rand(ndim)/sqrt(omega_true) for i in range(nwalkers)]
for p in p_0:
    p[-1] = log(exp(omega_true + omega_true*0.5*np.random.uniform(0, 1)) - 1.)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[zeta, y])
sampler.run_mcmc(p_0, 2000)
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

plt.hist(np.log(1. + np.exp(samples[:, 20])), histtype='step', normed=True, bins=20)

ax = plt.axis()
omega_min = max(ax[0], 0.)
omega_max = ax[1]

nrm = integrate.quad(ana_post, a=omega_min, b=omega_max, args=(zeta, sum_y2, n))
omega = np.linspace(omega_min, omega_max, 200)
post = np.zeros(len(omega))
for i in range(len(omega)):
    post[i] = ana_post(omega[i], zeta, sum_y2, n)

plt.plot(omega, post/nrm[0], label='analytical')
ax = plt.axis()
plt.plot([omega_true, omega_true], [ax[2], ax[3]], label='fiducial')
plt.ylim(ax[2], ax[3])
plt.legend(loc=0)
plt.show()

