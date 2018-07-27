import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import integrate
np.random.seed(4)
random.seed(4)
from my_hmc import simple_hmc,leapfrog,tempering,adaptive_hmc
from scipy.stats import multivariate_normal
from scipy.stats import norm

D=1
mass=np.identity(D)
inv_mass = np.identity(D)

def U(q):
    res = np.zeros(q.shape[0])
    for i in range(q.shape[0]):
        res[i] = - np.log(1 / 3 * np.exp(-(q[i] + 3) ** 2 / 2) + 2 / 3 * np.exp(-(q[i] - 1) ** 2 / (0.5) ** 2)+1e-5)

    return res


def K(p, inv_mass):
    res = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        res[i] = np.dot(np.dot(p[i], inv_mass), p[i]) / 2
    return res
grad_U = lambda q:-1/(U(q))*((1/3)*np.exp(-((q+3)**2)/2)*(-q-2) + (2/3)*np.exp(-((q-1)**2)/2)*(q+1))



x = np.linspace(-5, 5, 500)
y = lambda x:np.exp(-U(x))/1.4263607085121721
prop_y = norm(0,1).pdf(x)
plt.plot(x, y(x),label = "Target")
plt.plot(x,prop_y,label="Prior")
plt.legend()
plt.show()

x0 = multivariate_normal(np.zeros(D),mass).rvs().reshape(1,D)
q_hist = simple_hmc(U, K, grad_U, mass, inv_mass, 2000, x0, leapfrog)
q_hist = [q_hist[i].tolist()[0] for i in range(len(q_hist))]
plt.figure(figsize=(8, 8))
plt.subplot(2,1,1)
plt.hist(q_hist, bins=50, density=True)
plt.title("Simple HMC")
plt.plot(x, y(x),label = "Target")
plt.plot(x,prop_y,label = "Prior")
plt.legend()


q_hist = adaptive_hmc(U,K,grad_U,mass,inv_mass,2000,x0,leapfrog)
q_hist = [q_hist[i].tolist()[0] for i in range(len(q_hist))]

plt.subplot(2,1,2)
plt.hist(q_hist, bins=50, density=True)
plt.title("Adaptive HMC")
plt.plot(x, y(x),label = "Target")
plt.plot(x,prop_y,label = "Prior")
plt.legend()

plt.show()