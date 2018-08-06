import seaborn as sns
import autograd.numpy as np
import matplotlib.pyplot as plt
import random
from scipy import integrate

from my_hmc import simple_hmc,leapfrog,tempering,adaptive_hmc
from scipy.stats import multivariate_normal
from autograd import grad
#
mean1, cov1 = np.array([1, 0]), np.array([(0.1, .0), (.0, .1)])
mean2, cov2 = np.array([-3,0]), np.array([[0.1,0],[0,0.1]])


x1 = np.random.multivariate_normal(mean1, cov1, size=500)
x2 = np.random.multivariate_normal(mean2, cov2, size=500)

inv_cov1 = np.linalg.inv(cov1)
inv_cov2 = np.linalg.inv(cov2)

x = np.append(x1,x2,axis=0)

def bivariate_normal(X, Y, mean,cov):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.

    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """

    sigmax = cov[0][0]
    sigmay = cov[1][1]
    sigmaxy = cov[0][1]

    mux = mean[0]
    muy = mean[1]

    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom



######################################
######################################
######################################
######################################
D = 2
mass=np.identity(D)
inv_mass = np.identity(D)

def U(q):
    # res = np.zeros(q.shape[0])
    # for i in range(q.shape[0]):
    #     res[i] = - np.log(1 / 2 * np.exp( - np.dot(np.dot(q[i] - mean1,inv_cov1), q[i] - mean1)/2 )  + 1 / 2 * np.exp( - np.dot(np.dot(q[i] - mean2,inv_cov2), q[i] - mean2)/2 )+1e-5)
    # return res
    q = q.reshape(D,)
    return -np.log(1 / 2 * np.exp(- np.dot(np.dot(q - mean1, inv_cov1), q - mean1) / 2) + 1 / 2 * np.exp(
        - np.dot(np.dot(q - mean2, inv_cov2), q - mean2) / 2) + 1e-5)

def K(p, inv_mass):
    res = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        res[i] = np.dot(np.dot(p[i], inv_mass), p[i])/2
    return res

#grad_U = lambda q:-1/(U(q.reshape(D,)))*((1/2)*np.exp(- np.dot(np.dot(q.reshape(D,) - mean1,inv_cov1), q.reshape(D,) - mean1)/2 )*np.dot((-q.reshape(D,)+mean1),inv_cov1) + (1/2)*np.exp(- np.dot(np.dot(q.reshape(D,) - mean2,inv_cov2), q.reshape(D,) - mean2)/2 )*np.dot((-q+mean2),inv_cov2))
grad_U = grad(U)


# # compute True normalizer
# z = lambda q:np.exp(-U(q))
# Z = integrate.quad(z,float('-inf'),float('inf'))
# print(Z)


############################ visualize prior and true ##########################################
y = np.random.multivariate_normal(np.zeros(D), np.identity(D), size=500)
x = np.linspace(-4.0, 2.0,1000)
y = np.linspace(-1.0, 1.0,1000)
X, Y = np.meshgrid(x, y)

Z = (1/2)*bivariate_normal(X, Y, mean1,cov1) + (1/2)*bivariate_normal(X, Y, mean2,cov2)
z = bivariate_normal(X, Y, np.zeros(D),np.identity(D))
ax1= plt.contour(X,Y,Z)
#ax2 = plt.contour(X,Y,z)
plt.show()






x0 = multivariate_normal(np.zeros(D),mass).rvs().reshape(1,D)
# mass=np.identity(D)
# inv_mass = np.identity(D)
q_hist = simple_hmc(U, K, grad_U, mass, inv_mass, 100, x0, leapfrog,L=5,eps=0.075)


q_hist = np.asarray(q_hist)

samples = int(q_hist.shape[0]*3/4)

ax1= plt.contour(X,Y,Z)
plt.plot(q_hist[samples:, 0], q_hist[samples:, 1])
plt.xlim(-4,2)
plt.ylim(-1,1)
plt.show()


