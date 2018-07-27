

import autograd.numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from scipy.misc import logsumexp
from scipy import integrate
np.random.seed(14)
random.seed(14)
import seaborn as sns
from autograd import grad



def U(q):
    '''
    Potential Function: minus log unormalized density of interest

    :param q: the state
    :return:
    '''
    res = np.zeros(q.shape[0])
    for i in range(q.shape[0]):
        res[i] = np.dot(q[i]+3, q[i]+3) / 2

    return res


def K(p, inv_mass):
    res = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        res[i] = np.dot(np.dot(p[i], inv_mass), p[i]) / 2
    return res




def adaptive_metric(q_hist):
    '''
    :param: a list of vectors in which each vector is a sample from estimated target
    Approximate empirical covariance matrix of target distribution as mass matrix for kinetic distribution

    Note that the inverse metric more resembles the covariances of the target dist we will have a more uniform
    energy level set so easier exploration

    Once we are in typical set we run the Markov Chain using a default metric for a short window to build up
    initial estimate of the target covariance, then update the metric accordingly


    Particularly useful when Kinetic distribution is badly propsoed
    :return:
    '''
    q_hist = np.asarray(q_hist)
    cov = np.cov(q_hist.T)
    return cov




def leapfrog(U, K, grad_U, cur_q, mass,inv_mass, eps=0.01, L=200):
    '''
    Neal's proposed integrator
    '''
    D = cur_q.shape[1]

    q = cur_q.copy().reshape(1,D)

    p_0 = multivariate_normal(np.zeros(D),mass).rvs().reshape(1,D) # first sample momentum independently

    p = p_0.copy()
    # make a small step towords mode at beginning
    p = p - eps*grad_U(q)/2 #p = p - eps*grad_U(q.reshape(D,))/2

    # do full step
    for i in range(L):
        q += eps*p
        if i != L-1:
            p -= eps*grad_U(q)

    # Make a half step for momentum at the end.
    p = p-eps*grad_U(q)/2
    p = -p

    cur_U = U(cur_q.reshape(1,D))
    cur_K = K(p_0.reshape(1,D),inv_mass)
    proposed_U = U(q.reshape(1,D))
    proposed_K = K(p.reshape(1,D),inv_mass)

    if np.log(np.random.uniform()) < (cur_U - proposed_U + cur_K - proposed_K):
        return q
    return cur_q



def simple_hmc(U, K, grad_U, mass,inv_mass,iters, q_0, integrator):
    D = q_0.shape[1]
    q_hist = []
    q_hist.append(q_0.reshape(D,))
    accepted_num = 0
    cur_q = q_0.copy()


    for i in range(iters):

        nxt_q = integrator(U, K, grad_U, cur_q, mass,inv_mass, L=200, eps=0.05)

        if np.any(nxt_q !=cur_q):
            accepted_num += 1

            q_hist.append(np.asarray(nxt_q.reshape(D,)))
        cur_q = nxt_q

        if i%50 == 0:
            print("progressed {}%".format(i*100/iters))

    print("The acceptance rate is {}".format(accepted_num/iters))
    #corrplot(np.asarray(q_hist))
    return q_hist

def adaptive_hmc(U, K, grad_U, mass,inv_mass,iters, q_0, integrator):
    D = q_0.shape[1]
    q_hist = []
    q_hist.append(q_0.reshape(D,))
    accepted_num = 0
    cur_q = q_0.copy()

    for i in range(iters):
        nxt_q = integrator(U, K, grad_U, cur_q, mass,inv_mass, L=200, eps=0.05)

        if np.any(nxt_q != cur_q):
            accepted_num += 1
            q_hist.append(np.asarray(nxt_q.reshape(D,)))
        cur_q = nxt_q
        if i % 50 == 0:
            print("progressed {}%".format(i * 100 / iters))
        if i % 1000 and len(q_hist) > 200:
            # every 1000 iterations, we re-estimate the covariance of estimated target
            mass = adaptive_metric(q_hist[-200:])
            inv_mass = np.linalg.inv(mass + np.identity(D) * 1e-5)
    print("The acceptance rate is {}".format(accepted_num / iters))
    #corrplot(np.asarray(q_hist))
    return q_hist


def tempering(U, K, grad_U, q_0, mass, inv_mass, eps=0.05, L=200, temper=1.1):
    D = q_0.shape[1]

    p_0 = multivariate_normal(np.zeros(D), mass).rvs().reshape(1, D)

    q = q_0.copy()
    p = p_0.copy()

    U_initial = U(q.reshape(1,D))
    K_intial = K(p.reshape(1,D),inv_mass)

    sqrt_temper = np.sqrt(temper)

    traj = {'q':[],'p':[],'-H':[]}

    for i in range(L):
        if i < L/2:
            p = p * sqrt_temper
        else:
            p = p / sqrt_temper

        # leapfrog
        p = p - (eps / 2) * grad_U(q)
        q = q + eps * p
        p = p - (eps / 2) * grad_U(q)

        if i >= L/2:
            p = p / sqrt_temper
        else:
            p = p * sqrt_temper

        # save results for each jump
        traj["q"].append(q)
        traj["p"].append(p)


        U_cur = U(q.reshape(1,D))

        H = K(p.reshape(1,D),inv_mass) + U_cur
        traj["-H"].append(-H)

    p = -p
    traj["-H"] = np.asarray(traj["-H"])
    U_final = U_cur
    K_final = K(p.reshape(1,D),inv_mass)

    # propose the last state and either accept it or reject it

    H_initial = U_initial+K_intial
    H_prop = U_final+K_final


    if np.log(np.random.uniform()) < (-H_prop + H_initial):
        return q
    return q_0


def corrplot(trace,  maxlags=100):

    trace = trace[:,0]

    plt.acorr(trace-np.mean(trace),  normed=True, maxlags=maxlags)
    plt.xlim([0, maxlags])
    plt.show()





