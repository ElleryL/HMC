import numpy as np

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import uniform
import matplotlib.pyplot as plt
import random
np.random.seed(447)
random.seed(447)
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import annealing_importance_sampling as ais

######### TOY #########
D = 1
f_0 = lambda x:(1/3*multivariate_normal(np.zeros(D),np.diag(np.ones(D)*0.1)).pdf(x) \
               + 2/3*multivariate_normal(np.ones(D)*-5,np.diag(np.ones(D)*0.05)).pdf(x))
#
#
# # # ########### baseline test for multimode ########### Set D = 1 test on mean, variance
f_0_mean = lambda x:x*(1/3*multivariate_normal(np.zeros(D),np.diag(np.ones(D)*0.1)).pdf(x) \
               + 2/3*multivariate_normal(np.ones(D)*-5,np.diag(np.ones(D)*0.05)).pdf(x))

true_mean = integrate.quad(f_0_mean,float("-inf"),float('inf'))[0]

x = np.linspace(-10,10,1000)

#
# for i in range(len(beta)):
#     y = f_0(x)**(beta[i])*norm(0,1).pdf(x)**(1-beta[i])
#     plt.plot(x,y)
# plt.show()

####### Testing on some examples that importance sampling performs worse than AIS works when the original proposed sampler is not good ##############
func_in_expect = lambda x:(x)
p_n = multivariate_normal(np.zeros(D),np.diag(np.ones(D))*2)
beta = np.linspace(0.,1,200)

beta_prime = np.linspace(0,0.5,50)
beta_prime = np.append(beta_prime,np.linspace(0.5,1,150))  # tuned for challenge case
beta=beta_prime
res = ais.annealing(p_n, f_0, ais.proposed_sampler, beta, func_in_expect, mc_iters=1, samples_size=3000, D=D)

plt.plot(res,label = 'AIS')
print(">>>>>>>>>")
beta = np.linspace(0,1,2)
res = ais.annealing(p_n, f_0, ais.proposed_sampler, beta, func_in_expect, mc_iters=0, samples_size=3000, D=D)

plt.plot(res,label = 'IS')
plt.axhline(y=true_mean,linewidth=4, color='r',label = 'True')
plt.legend()
plt.show()
y = f_0(x)
yy = multivariate_normal(np.zeros(D),np.diag(np.ones(D)*2)).pdf(x)
plt.plot(x,y,label = 'Posterior')
plt.plot(x,yy,label = 'Prior')
plt.legend()
plt.show()

print("The true value is {}".format(true_mean))

