import numpy as np
import matplotlib.pyplot as plt
import emcee
from covariance_estimation.analytical_covolution import (log_posterior_jeffreys_prior, posterior_jeffreys_prior)
from scipy import integrate
from math import sqrt  # , log10
import pymc3

np.random.seed(31415)

zeta = 1.
sum_y2 = 43.9599768767
n = 20
omega_true = 1.

# find the roots to the quadratic equation for the MAP
a = n+2.
b = (n+4.)*zeta - sum_y2
c = 2*zeta**2
print('b = ', b)
delta = b**2 - 4.*a*c
print('delta = ', delta)
omega_bar_1 = None
omega_bar_2 = None
if delta >= 0:
    omega_bar_1 = (-b + sqrt(delta)) / 2. / a
    omega_bar_2 = (-b - sqrt(delta)) / 2. / a
    print('real roots are ', omega_bar_1, 'and', omega_bar_2)

# use emcee to sample from the posterior
ndim = 1
nwalkers = 10
p_0 = [(1+np.random.rand(ndim)) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_jeffreys_prior, args=[zeta, sum_y2, n])
sampler.run_mcmc(p_0, 2000)
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

# plot the histogram of samples
plt.hist(samples, histtype='step', normed=True, bins=20)
omega_est_mean = np.mean(samples)
omega_est_hpd = pymc3.stats.hpd(samples)
print('omega_est_hpd = ', omega_est_hpd)
ax = plt.axis()
omega_min = max(ax[0], 0)
omega_max = ax[1]
print('omega_min = ', omega_min)
plt.clf()
plt.hist(samples, histtype='step', normed=True, bins=30, range=(0, omega_max), label='samples')

# omega_min = 1e-2
# omega_max = 5
# omega = np.logspace(log10(omega_min), log10(omega_max), 100)
# log_post = np.zeros(len(omega))
# for i in range(len(omega)):
#     log_post[i] = log_posterior_jeffreys_prior(omega[i], zeta, sum_y2, n)
# plt.plot(omega, log_post)

# find the normalised analytical posterior
nrm = integrate.quad(posterior_jeffreys_prior, a=omega_min, b=omega_max, args=(zeta, sum_y2, n))
omega = np.linspace(omega_min, omega_max, 500)
post = np.zeros(len(omega))
for i in range(len(omega)):
    post[i] = posterior_jeffreys_prior(omega[i], zeta, sum_y2, n)

# plot the analytical posterior and the MAP(s)
# one of the will be a trough and the other a peak
plt.plot(omega, post/nrm[0], label='analytical')
ax = plt.axis()
if omega_bar_1 is not None:
    plt.plot([omega_bar_1, omega_bar_1], [ax[2], ax[3]], label='MAP-1')
if omega_bar_2 is not None:
    plt.plot([omega_bar_2, omega_bar_2], [ax[2], ax[3]], label='MAP-2')

# plot the mean and the true values of omega
plt.plot([omega_est_mean, omega_est_mean], [ax[2], ax[3]], label='mean')
plt.plot([omega_true, omega_true],[ax[2], ax[3]], label='true')
if len(omega_est_hpd) == 1:
    left_line = omega_est_hpd[0][0]
    right_line = omega_est_hpd[0][1]
    plt.fill_betweenx(y=np.asarray([ax[2], ax[3]]), x1=np.asarray([left_line, left_line]),
                      x2=np.asarray([right_line, right_line]), label='hpd', alpha=0.1)
plt.ylim([ax[2], ax[3]])
plt.xlim(omega_min, omega_max)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$p(\omega)$')
plt.legend()
plt.show()
