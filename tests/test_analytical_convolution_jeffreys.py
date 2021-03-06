import numpy as np
import matplotlib.pyplot as plt
import emcee
from covariance_estimation.analytical_covolution import (log_posterior_jeffreys_prior, posterior_jeffreys_prior)
from scipy import integrate
import seaborn
from math import sqrt
import pymc3

np.random.seed(31415)

ndim = 1
nwalkers = 10

zeta = 1.
sum_y2 = 21.9802082382
n = 20
omega_true = 1e-2
a = n+1.
b = (n+2.)*zeta - sum_y2
c = zeta**2
print('b = ', b)
delta = b**2 - 4.*a*c
print('delta = ', delta)
if delta >= 0:
    omega_bar_1 = (-b + sqrt(delta)) / 2. / a
    omega_bar_2 = (-b - sqrt(delta)) / 2. / a
    print('real roots are ', omega_bar_1, 'and', omega_bar_2)

p_0 = [(1+np.random.rand(ndim)) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_jeffreys_prior, args=[zeta, sum_y2, n])
sampler.run_mcmc(p_0, 2000)
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

plt.hist(samples, histtype='step', normed=True, bins=20)
omega_est_mean = np.mean(samples)
omega_est_hpd = pymc3.stats.hpd(samples)
print('omega_est_hpd = ', omega_est_hpd)
ax = plt.axis()
omega_min = max(ax[0], 0.)
omega_max = ax[1]
plt.clf()
plt.hist(samples, histtype='step', normed=True, bins=30, range=(0, omega_max), label='samples')

nrm = integrate.quad(posterior_jeffreys_prior, a=omega_min, b=omega_max, args=(zeta, sum_y2, n))
omega = np.linspace(omega_min, omega_max, 200)
post = np.zeros(len(omega))
for i in range(len(omega)):
    post[i] = posterior_jeffreys_prior(omega[i], zeta, sum_y2, n)

plt.plot(omega, post/nrm[0], label='analytical')
ax = plt.axis()
# plt.plot([omega_bar, omega_bar], [ax[2], ax[3]])
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
