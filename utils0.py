'''
Compare Simple HMC verse Adaptive HMC
'''

import autograd.numpy as np
import matplotlib.pyplot as plt
import random
from scipy import integrate
from scipy.stats import norm
np.random.seed(4)
random.seed(4)
from my_hmc import simple_hmc,adaptive_hmc,leapfrog,corrplot,K,U,tempering
from scipy.stats import multivariate_normal
from autograd import grad





grad_U = lambda q:q+3





######################################## visualizer ################################################################

#plot_density(U)
x = np.linspace(-10, 10, 1000)
y = lambda x:np.exp(-U(x))/2.5066282746310002
prop_y = norm(0,.5**(1/2)).pdf(x)
plt.plot(x, y(x),label = "Target")
plt.plot(x,prop_y,label="Prior")
plt.legend()
plt.show()





########################### Run ########################################################################################

D = 1
mass=np.identity(D)*0.5
inv_mass = np.identity(D)*2




x0 = multivariate_normal(np.zeros(D),np.identity(D)).rvs().reshape(1,D)

q_hist = simple_hmc(U,K,grad_U,mass,inv_mass,5000,x0,leapfrog)
q_hist_adaptive = adaptive_hmc(U, K, grad_U, mass,inv_mass,5000, x0, leapfrog)




################################ 1D Histogram Plot ################################################



q_hist = [q_hist[i].tolist()[0] for i in range(len(q_hist))]
q_hist_adaptive = [q_hist_adaptive[i].tolist()[0] for i in range(len(q_hist_adaptive))]
#
plt.figure(figsize=(8, 8))
plt.subplot(2,1,1)
plt.hist(q_hist, bins=50, density=True)
plt.title("Simple HMC")
plt.plot(x, y(x),label = "Target")
plt.plot(x,prop_y,label = "Prior")
plt.legend()
#
# #
plt.subplot(2,1,2)
plt.hist(q_hist_adaptive, bins=50, density=True)
plt.title("adaptive HMC")
plt.plot(x, y(x),label = "Target")
plt.plot(x,prop_y,label = "Prior")
plt.legend()
plt.show()

