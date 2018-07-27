
import autograd.numpy as np
import matplotlib.pyplot as plt
import random
from scipy import integrate
from scipy.stats import norm
np.random.seed(4)
random.seed(4)
from my_hmc import simple_hmc,adaptive_hmc,leapfrog,corrplot,K,tempering
from scipy.stats import multivariate_normal
from autograd import grad



def U(q):


    z1, z2 = q[:,0],q[:,1]
    norm = np.sqrt(z1 ** 2 + z2 ** 2)

    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)

    u = 0.5 * ((norm - 4) / 0.4) ** 2 - np.log(exp1 + exp2 + 1e-5)

    return u


grad_U = grad(U)

def plot_density(density):

    X_LIMS = (-7, 7)
    Y_LIMS = (-7, 7)

    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    x1, x2 = np.meshgrid(x1, x2)
    shape = x1.shape
    x1 = x1.ravel()
    x2 = x2.ravel()

    z = np.c_[x1, x2]


    density_values = density(z).reshape(shape)
    density_values = np.exp(-density_values)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(density_values, extent=(*X_LIMS, *Y_LIMS), cmap="summer")
    ax.set_title("True density")
    plt.show()
    plt.close()


def scatter_points(points,title,ax,density):

    X_LIMS = (-7, 7)
    Y_LIMS = (-7, 7)

    ax.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25)

    # ax.set_xlim(np.min(points[:, 0]),np.max(points[:, 0]))
    # ax.set_ylim(np.min(points[:, 1]),np.max(points[:, 1]))

    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    x1, x2 = np.meshgrid(x1, x2)
    shape = x1.shape
    x1 = x1.ravel()
    x2 = x2.ravel()

    z = np.c_[x1, x2]


    density_values = density(z).reshape(shape)
    density_values = np.exp(-density_values)

    ax.set_xlim(*X_LIMS)
    ax.set_ylim(*Y_LIMS)

    ax.imshow(density_values, extent=(*X_LIMS, *Y_LIMS), cmap="summer")

    ax.set_title(
        "{}"
        .format(title)
    )



D = 2
mass=np.identity(D)
inv_mass = np.identity(D)
plot_density(U)
x0 = multivariate_normal(np.zeros(D),np.identity(D)).rvs().reshape(1,D)
q_hist = simple_hmc(U,K,grad_U,mass,inv_mass,500,x0,leapfrog)
q_hist_adaptive = adaptive_hmc(U,K,grad_U,mass,inv_mass,500,x0,leapfrog)

plt.figure(figsize=(8, 8))
ax = plt.subplot(2,1,1)
scatter_points(np.asarray(q_hist),"Simple HMC",ax,U)
ax = plt.subplot(2,1,2)
scatter_points(np.asarray(q_hist_adaptive),"Adaptive HMC",ax,U)
plt.show()
plt.close()
