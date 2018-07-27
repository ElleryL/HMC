
'''

Bayesian Linear Regression

params[0] = log(sigma_error)   Normal(0,4)
params[1] = log(sigma_beta)    Normal(0,4)
params[2] = interception parameter Normal(0,100)
params[3:] = weights Normal(0,sigma_beta**2)

'''

import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
import random
np.random.seed(447)
random.seed(447)

data1 = np.loadtxt('data1.txt') #toy data experiments
y = data1[:,0]
x = data1[:,1:]

prior_mu = np.array([0,0,0])
prior_sig = np.array([2,2,10])

def prior_dist(params):
    return np.sum(norm(prior_mu,prior_sig).logpdf(params[:3])) + np.sum(norm(0,np.exp(params[1])).logpdf(params[3:]))

def likelihood(params):
    pred = params[2] + np.dot(x, params[3:])
    return np.sum(norm(pred, np.exp(params[0])).logpdf(y))


def unnormalized_log_posterior(params):
    return likelihood(params) + prior_dist(params)

def grad_unnormalized_log_posterior(params):
    grad_prior = -params[:3]/prior_sig**2
    grad_prior[1] += -2+np.sum(params[3:]**2)*np.exp(-2*params[1])
    grad_prior = np.append(grad_prior,-params[3:]*np.exp(-2*params[1]))

    pred = params[2] + np.dot(x, params[3:])
    grad = np.zeros(params.shape[0])
    N = pred.shape[0]
    grad[0] = -N + np.sum((pred-y)**2)*np.exp(-2*params[0])
    grad[2] = np.sum(-(pred-y)/np.exp(params[0])**2)
    grad[3] = (-np.sum((pred-y)*x[:,0])/np.exp(params[0])**2)
    grad[4] = (-np.sum((pred-y) * x[:, 1]) / np.exp(params[0]) ** 2)
    grad = grad + grad_prior
    return grad

def langevin_sampler(unnormalized_target,iters,step_size,init_params):
    cur_params = init_params
    D = init_params.shape[0]
    params = np.zeros((iters, D))
    accepted = np.zeros(D)
    for i in range(iters):
        params[i] = cur_params
        for j in range(D):
            proposed_params = params[i].copy()
            grad = grad_unnormalized_log_posterior(proposed_params)

            proposed_params[j] = norm(proposed_params[j] + step_size[j]*grad[j], step_size[j]).rvs()
            if (uniform().rvs() < np.exp(unnormalized_target(proposed_params) - unnormalized_target(params[i]))):
                accepted[j] += 1
                params[i] = proposed_params
        cur_params = params[i]
    print("Accpete Rate {}".format(accepted / iters))
    return params

def metropolist_hastings(unnormalized_target, iters, step_size,init_params,component_update):
    cur_params = init_params
    D = init_params.shape[0]
    params = np.zeros((iters,D))
    if not component_update: # update all components at once
        accepted = 0
        for i in range(iters):
            params[i] = cur_params
            proposed_params = params[i].copy()
            for j in range(D):
                proposed_params[j] = norm(proposed_params[j],step_size[j]).rvs()
            if (uniform().rvs() < np.exp(unnormalized_target(proposed_params) - unnormalized_target(params[i]))):

                accepted += 1
                params[i] = proposed_params
            cur_params = params[i]
        print("Accepted Rate {}".format(accepted/iters))
    else:
        accepted = np.zeros(D)
        for i in range(iters): # update single component Metropolis Hasting
            params[i] = cur_params
            for j in range(D):
                proposed_params = params[i].copy()
                proposed_params[j] = norm(proposed_params[j],step_size[j]).rvs()
                if (uniform().rvs() < np.exp(unnormalized_target(proposed_params) - unnormalized_target(params[i]))):
                    accepted[j] += 1
                    params[i] = proposed_params
            cur_params = params[i]
        print("Accpete Rate {}".format(accepted/iters))
    return params




def error(x,y,weights):
    x = np.column_stack((np.ones(x.shape[0]), x))
    err = np.mean((np.dot(x,weights) - y)**2)
    return err

def data_visualizer(x,y):

    D = x.shape[1]
    for i in range(D):
        plt.subplot(2,1,i+1)
        plt.scatter(x[:,i],y)
        plt.xlabel("x{}".format(i+1))
        plt.ylabel("y")
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()
    plt.close()

############################ Compute Frequenciest Linear Regression As Baseline  ########################################################

def simple_LR(x,y):
    x = np.column_stack((np.ones(x.shape[0]), x))
    z = np.dot(x.T,x)
    xy = np.dot(x.T,y)
    z = np.linalg.solve(z,xy)

    return z


###################################### Visualization ######################################

def param_hist(samples, mle_weights):
    D = samples.shape[1]
    plt.figure(figsize=(10, 10))

    for i in range(2,D):
        ax = plt.subplot(D,2,i+1)
        plt.axvline(x=mle_weights[i-2],linewidth=4, color='r')
        sns.distplot(samples[:,i],kde=False,ax=ax)
        _=ax.set(title='historgram of parameters beta{} with Metropolis Hasting'.format(i))
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()
    plt.close()

def param_curve(samples, mle_weights):
    D = samples.shape[1]
    plt.figure(figsize=(10, 10))
    for i in range(2,D):
        ax = plt.subplot(D,2,i+1)
        plt.axhline(y=mle_weights[i-2],linewidth=4, color='r')
        plt.plot(samples[:,i])
        _=ax.set(title='parameters beta{} with Metropolis Hasting'.format(i))
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()
    plt.close()
mle_weights = simple_LR(x,y)

step_size = np.array([0.4,0.8,0.1,0.1,0.1]) # hyperparameters
init_params = np.zeros(5)
#res = metropolist_hastings(unnormalized_log_posterior, 1000, step_size, init_params, component_update=True)
res = langevin_sampler(unnormalized_log_posterior,1000,step_size,init_params)  # slow due to compute gradient and accept rate is low

samples = res[500:,:]
weights = np.mean(samples,axis=0)

print("The prediction error for Bayesian Regression is {}".format(error(x,y,weights[2:])))
param_hist(samples, mle_weights)
param_curve(res, mle_weights)


####################################################################################################
