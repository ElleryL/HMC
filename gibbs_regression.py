import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 5)


def sample_beta(y, x, beta, tau_y, mu, tau):

    D = beta.shape[0]
    z = y - np.dot(x,beta)
    result = np.zeros(D)
    for j in range(D):
        precision = tau[j] + tau_y * np.sum(x[:,j] * x[:,j])
        mean = tau[j] * mu[j] + tau_y * np.sum((z + beta[j]*x[:,j]) * x[:,j])
        mean /= precision
        result[j] = np.random.normal(mean, 1 / np.sqrt(precision))
    return result

def sample_tau(y, x, beta, a, b):
    N = len(y)
    alpha_new = a + N / 2
    resid = y - np.dot(x,beta)
    beta_new = b + np.sum(resid * resid) / 2
    return np.random.gamma(alpha_new, 1 / beta_new)

# N = 50
# x = np.random.uniform(low = 0, high = 4, size = N)
# y = np.random.normal(beta_0_true + beta_1_true * x, 1 / np.sqrt(tau_true))

# synth_plot = plt.plot(x, y, "o")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()


data1 = np.loadtxt('data1.txt')
y = data1[:,0]
x = data1[:,1:]



init = {"beta": np.zeros(3),
        "tau_y": 2}

## specify hyper parameters
hypers = {"mu": np.zeros(3),
         "tau": np.ones(3),
         "a": 2,
         "b": 1}

def gibbs_sampler(y, x, iters, init, hypers):

    beta = init["beta"]
    tau_y = init["tau_y"]

    trace = np.zeros((iters, 4))  ## trace to store values of beta_0, beta_1, tau

    for i in range(iters):
        beta = sample_beta(y, x, beta, tau_y, hypers["mu"], hypers["tau"])
        tau_y = sample_tau(y, x, beta, hypers["a"], hypers["b"])
        trace[i, :] = np.append(beta,tau_y)

    # trace = pd.DataFrame(trace)
    # trace.columns = ['beta_0', 'beta_1','beta_2', 'tau']

    return trace


iters = 2000
x = np.column_stack((np.ones(x.shape[0]), x))
trace = gibbs_sampler(y, x, iters, init, hypers)


def simple_LR(x,y):
    z = np.dot(x.T,x)
    xy = np.dot(x.T,y)
    z = np.linalg.solve(z,xy)

    return z

mle_weights = simple_LR(x,y)

def visualization(samples,trace,mle_weights):
    D = samples.shape[1]

    plt.figure(figsize=(7, 8))
    for i in range(D-1):
        ax = plt.subplot(3,1,i+1)
        plt.axvline(x=mle_weights[i],linewidth=4, color='r')
        sns.distplot(samples[:,i],kde=False,ax=ax)
        _=ax.set(title='historgram of parameters beta{} with Gibbs'.format(i))
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()
    plt.close()

    D = trace.shape[1]
    plt.figure(figsize=(7, 8))
    for i in range(D-1):
        ax = plt.subplot(3,1,i+1)
        plt.axhline(y=mle_weights[i],linewidth=4, color='r')
        plt.plot(trace[:,i])
        _=ax.set(title='parameters beta{} with Gibbs'.format(i))
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()
    plt.close()

visualization(trace[1500:],trace, mle_weights)

    # traceplot = trace.plot()
    # traceplot.set_xlabel("Iteration")
    # traceplot.set_ylabel("Parameter value")
    # plt.show()
    #
    # trace_burnt = trace[1500:1999]
    # hist_plot = trace_burnt.hist(bins = 30, layout = (1,4))
    # plt.show()
    #
    # print(trace_burnt.median())
    # print(trace_burnt.std())