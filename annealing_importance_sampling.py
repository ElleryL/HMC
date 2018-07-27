import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random
np.random.seed(447)
random.seed(447)




def proposed_sampler(D):
    '''
    Generic proposed multi-dimensional distribution
    Proposal distribution: 1/Z * f_n
    :return:
    '''
    mu = np.zeros(D)
    var = np.diag(np.ones(D)*2)

    return multivariate_normal(mu,var)


def intermediate_sampler(f_0, f_n, beta_j, D, x):
    '''
    Intermidiate distribution between target f_0 and initial proposed f_n
    :return:
    '''

    return f_0(x) ** beta_j * f_n(D).pdf(x) ** (1 - beta_j)

def random_transition(f_0,f_n,x,beta_j,step_size,mc_iters,D):
    cur_x = x
    for i in range(mc_iters):
        proposed_x = multivariate_normal(cur_x,step_size).rvs()
        accept = intermediate_sampler(f_0, f_n, beta_j, D, proposed_x) / intermediate_sampler(f_0, f_n, beta_j, D, cur_x)
        if np.random.rand() < accept:
            cur_x = proposed_x
    return cur_x

def annealing(p_n,f_0,f_n,beta,func_in_expect,mc_iters=5,samples_size=1000,D=6):
    '''
    :param p_n: p_n = (1/Z)*f_n; proposed probability distribution that we can sampled
    :return:
    '''
    samples=np.zeros((samples_size,D))
    weights = np.zeros(samples_size)
    step_size = np.diag(np.ones(D)*0.5)

    hist = []

    for i in range(samples_size):
        cur_x = p_n.rvs()
        w = 1
        for j in range(1,beta.shape[0]):
            cur_x = random_transition(f_0,f_n,cur_x, beta[j], step_size, mc_iters,D)
            w += np.log(intermediate_sampler(f_0, f_n, beta[j], D, cur_x)) - np.log(intermediate_sampler(f_0, f_n, beta[j - 1], D, cur_x))
        samples[i] = cur_x
        weights[i] = np.exp(w)

        g = func_in_expect(samples[:i+1,:])
        hist.append(1 / np.sum(weights) * np.sum(g * weights[:i+1, None], axis=0))

        if i%100 == 0:
            #print("currently have progressed {}%".format(i/samples_size*100))
            print(hist[-1])

        g = func_in_expect(samples)
    # TODO: Compute normalizing constant;
    # plt.plot(hist)
    # plt.show()
    print("weight log variance {}".format(np.log(np.var(weights))))
    res = 1/np.sum(weights) * np.sum(g*weights[:,None],axis=0)
    print(res)
    hist.append(res)

    return hist
















