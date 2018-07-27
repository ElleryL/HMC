'''
Code for all figure and graph in report

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
import seaborn as sns
import pandas as pd



x = np.linspace(-3,3,500)

plt.plot(x,np.zeros(500),c='k',alpha=0.5)
plt.scatter(x=1,y=0,c='r',s=50)
plt.text(3.5,-0.00005,"q",fontsize=15)
plt.xlim(-5,5)
plt.show()