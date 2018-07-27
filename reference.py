
import autograd.numpy as np
import matplotlib.pyplot as plt
import random
from scipy import integrate
from scipy.stats import norm
np.random.seed(4)
random.seed(4)
from my_hmc import simple_hmc,adaptive_hmc,leapfrog,corrplot,K,U
from scipy.stats import multivariate_normal
from autograd import grad

def U(q):


    z1, z2 = q[:,0],q[:,1]
    norm = np.sqrt(z1 ** 2 + z2 ** 2)

    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)

    u = 0.5 * ((norm - 4) / 0.4) ** 2 - np.log(exp1 + exp2 + 1e-5)

    return np.exp(-u)
