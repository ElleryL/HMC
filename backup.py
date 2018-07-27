
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from scipy.misc import logsumexp
from scipy import integrate
# np.random.seed(14)
# random.seed(14)
import seaborn as sns




################################ For visualization of contour################################

def walk_visual(trace_q ,trace_p):
    pass




###############################################################################################



def potential_U(q):
    '''
    minus the log probability densify of the distribution that we wish to sample (-log(target_distribution)) similar to posterior distribution
    :return:
    '''
    pass

def kinetic_K(p ,mass):
    '''
    log(proposed distribution) similar to prior distribution
    :return:
    '''
    pass





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
    cov = np.cov(q_hist)
    return cov

def construct_tractory(grad_U ,z0 ,direction ,L ,eps=0.01 ,mass=1):
    '''
    Constuct a tractory with given direction using some approximation integrator
    In this function we just use leapfrog
    :param z0: (p0,q0) position in extended dual parameter space
    :return:
    '''
    t = []
    p0 ,q0 = z0

    p = p0.copy()
    q = q0.copy()

    if direction == 1: # going forward
        p = p - eps *grad_U(q ) /2

        for i in range(L):
            q = q + eps * p /mass # change to inverse of matrix for high dimension later
            if i != L- 1:
                p = p - eps * grad_U(q)

                t.append(np.array([p, q]))

        p = p - eps * grad_U(q) / 2

        p = -p
        t.append(np.array([p, q]))
    else:
        p = p + eps * grad_U(q) / 2

        for i in range(L):
            # change to inverse of matrix for high dimension later
            q = q - eps * p / mass
            if i != L - 1:
                p = p + eps * grad_U(q)
                t.append(np.array([p, q]))
        p = p + eps * grad_U(q) / 2

        p = -p
        t.append(np.array([p, q]))

    return t


'''
Challenges : if the target distribution have multiple modes; some modes are isoldated. Hard to jump between modes
             Because H is conserved along a trajectory; once in typical region, unlikley pass through the low probability
             tunnel to another mode.

             Moving slow with vastly different probability density

Solution   : give up volumn persevation; so that acceptance rate invovles the Jacobian matrix (accounting for volumn change)
             Can use it as an approximator for normalizing constant of target distribution
'''


def simple_spiral_method(U, K, grad_U, cur_q, mass, eps=0.05, L=200, alpha=1.5):
    D = cur_q.shape[0]
    q = cur_q.copy()

    p_0 = multivariate_normal(np.zeros(D), mass).rvs()  # first sample momentum independently

    p = p_0.copy()
    # randomly select a state somewhere in the middle of integration time; if such state happend to be in the middle we have reversible
    l = np.random.randint(0, L)
    r = L - l

    trace_p = np.zeros((L, D))
    trace_q = np.zeros((L, D))

    trace_p[0] = p_0.copy()
    trace_q[0] = cur_q.copy()

    alpha_power = np.zeros(L)
    # Now do forward moves

    p = p * np.sqrt(alpha)
    p = p - eps * grad_U(q) / 2
    for i in range(1, l):
        # q += eps * p * np.sqrt(alpha)
        q += eps * p
        if i != l - 1:
            p -= eps * grad_U(q)
            trace_q[i] = q.copy()
            trace_p[i] = p.copy()
            alpha_power[i] = i
    # Make a half step for momentum at the end.
    p = p - eps * grad_U(q) / 2
    p = -p * np.sqrt(alpha)
    trace_q[l - 1] = q.copy()
    trace_p[l - 1] = p.copy()
    alpha_power[l - 1] = l - 1

    # Now do backward moves from original state
    q = cur_q.copy()
    p = p_0.copy()

    p = p / np.sqrt(alpha)
    p = p + eps * grad_U(q) / (2)

    k = -1
    for i in range(l, L):
        # q += eps * p / np.sqrt(alpha)
        q -= eps * p
        if i != L - 1:
            p += eps * grad_U(q)
            trace_q[i] = q.copy()
            trace_p[i] = p.copy()

            alpha_power[i] = k
            k -= 1
    # Make a half step for momentum at the end.
    p = p + eps * grad_U(q) / (2)
    p = -p / np.sqrt(alpha)

    trace_q[L - 1] = q.copy()
    trace_p[L - 1] = p.copy()

    alpha_power[L - 1] = k

    # we gonna random select a move from trace with probability proportional to prob(q,p)*alpha

    # neg_H = lambda trace,mass : - U(trace[:,0].reshape(trace.shape[0],D)) - K(trace[:,1].reshape(trace.shape[0],D),mass)

    neg_H = lambda trace_q, trace_p, mass: - U(trace_q) - K(trace_p, mass) + alpha_power + np.log(alpha)

    prob = neg_H(trace_q, trace_p, mass) - logsumexp(neg_H(trace_q, trace_p, mass))

    nxt_q_inx = np.random.choice(trace_p.shape[0], p=np.exp(prob.reshape(L, )))

    proposed_q = trace_p[nxt_q_inx]
    proposed_p = trace_p[nxt_q_inx]

    cur_U = U(cur_q.reshape(1, D))
    cur_K = K(p_0.reshape(1, D), mass)
    proposed_U = U(proposed_q.reshape(1, D))
    proposed_K = K(proposed_p.reshape(1, D), mass)

    if np.log(np.random.uniform()) < (cur_U - proposed_U + cur_K - proposed_K):
        return proposed_q

    return cur_q


def doubly_spiral_method(U, K, grad_U, cur_q, mass, eps=0.01, L=200, alpha=0.5):
    '''
    Simply extends the above method by randomly jumpy back and forth
    '''
    pass


def probabilistic_trajectory(U, K, grad_U, cur_q, mass, eps=0.01, L=200, W=100):
    '''

    :param W: windows size
    :return:
    '''

    D = cur_q.shape[0]
    q = cur_q.copy()

    p_0 = multivariate_normal(np.zeros(D), mass).rvs()  # first sample momentum independently
    p = p_0.copy()

    s = np.random.uniform(0, W)


def HIS(K, alpha, eps, T):
    '''
    Incoprate Hamiltonian into Importance Sampling


    TODO: this is similar to spiral method;

    :param: T: temperature
    :param: alpha: cool system so that T = 1
    :param: L: how long do we do trajectory
    :param: eps: stepsize

    z = (p,q)

    prob(z) proportional to exp(-H(p,q)/T)

    Step (1) :
    sample p uniformly over some region V as a start point
    sample momentum q proportional to exp(-K(q)) e.g. N(p,mass)

    Step (2) :
    Do some trajectory over the given energy level curve to some new point z' = (p',q')

    Step (3):
    Compute the weight = exp(-H(p',q')) / N(p,mass) * alpha**(K*D*V)

    Step (4):
    Approximate normalzing constant = np.mean(weights)

    :return:
    '''
    pass


def static_trajectory(U, K, grad_U, cur_q, mass, eps=0.01, L=200):
    '''
    A simple modification to this will make it into NUTS

    Dynamically construct trajectory until stop condition met
    where prob(z) is proportional to cannoical density to keep invariance property

    To satisfy the reversibility, we uniformly sample a tracjectory that contains the initial points

    '''

    D = cur_q.shape[0]
    q = cur_q.copy()
    p_0 = multivariate_normal(np.zeros(D), mass).rvs().reshape(D, )
    p = p_0.copy()

    H = lambda p, q, mass: K(p, mass) + U(q)  # energy level

    z = np.array([p, q])

    # Step(1) Construct an intial trajectory contains z0; starts with a single elements
    old_t = np.array([z])

    while len(old_t) < L:
        # Step(2) randomly select integration direction of forward or backward
        direction = np.random.binomial(1, 0.5)

        # sample tractory t' from Uniform(t|z') with larger sizer
        # step(3) append points p,q along this direction;
        new_t = construct_tractory(grad_U, z, direction, L=20, eps=0.01, mass=1)
        new_t = np.asarray(new_t)

        # step(4) sample  z' = (p',q') with bernoulli process
        # T(z'|t) = prob(t=old)T(z'|t = old) + prob(t=new)T(z'|t = new)

        Z_old = np.sum(np.exp(-H(old_t[:, 0], old_t[:, 1], mass)))
        Z_new = np.sum(np.exp(-H(new_t[:, 0], new_t[:, 1], mass)))

        prob_old = Z_old / (Z_old + Z_new)
        prob_new = Z_new / (Z_old + Z_new)

        # compute the distribution of z and conditional distribution z given trajectory

        # TODO: change to logsumexp for numerical stable
        T_z_given_old_t = np.exp(-H(old_t[:, 0], old_t[:, 1], mass)) / Z_old
        T_z_given_new_t = np.exp(-H(new_t[:, 0], new_t[:, 1], mass)) / Z_new

        T_z_given_t = np.append(prob_old * T_z_given_old_t, prob_new * T_z_given_new_t)

        # sampling z'
        new_t = np.append(old_t, new_t, axis=0)

        z = new_t[np.random.choice(new_t.shape[0], p=T_z_given_t)]
        old_t = new_t
        # step(5) repeat until length L is met
        # TODO : construct a more sophisticated stop condition based for different the energy level
    z = new_t[np.random.choice(new_t.shape[0], p=T_z_given_t)]

    proposed_q = z[0]
    proposed_p = z[1]

    cur_U = U(cur_q.reshape(1, D))
    cur_K = K(p_0.reshape(1, D), mass)
    proposed_U = U(proposed_q.reshape(1, D))
    proposed_K = K(proposed_p.reshape(1, D), mass)

    if np.log(np.random.uniform()) < (cur_U - proposed_U + cur_K - proposed_K):
        return proposed_q
    return cur_q


def leapfrog(U, K, grad_U, cur_q, mass, eps=0.01, L=200):
    '''
    Neal's proposed integrator
    '''
    D = cur_q.shape[0]

    q = cur_q.copy()
    p_0 = multivariate_normal(np.zeros(D), mass).rvs()  # first sample momentum independently

    p = p_0.copy()
    # make a small step towords mode at beginning
    p = p - eps * grad_U(q) / 2

    # do full step
    for i in range(L):
        q += eps * p
        if i != L - 1:
            p -= eps * grad_U(q)

    # Make a half step for momentum at the end.
    p = p - eps * grad_U(q) / 2
    p = -p

    cur_U = U(cur_q.reshape(1, D))
    cur_K = K(p_0.reshape(1, D), mass)
    proposed_U = U(q.reshape(1, D))
    proposed_K = K(p.reshape(1, D), mass)

    if np.log(np.random.uniform()) < (cur_U - proposed_U + cur_K - proposed_K):
        return q
    return cur_q


def hmc(U, K, grad_U, iters, q_0, integrator, adaptive):
    q_hist = []
    q_hist.append(q_0[0])
    accepted_num = 0
    cur_q = q_0.copy()

    mass = np.identity(q_0.shape[0]) * 0.5

    for i in range(iters):

        nxt_q = integrator(U, K, grad_U, cur_q, mass, L=200, eps=0.05)
        if np.all(nxt_q != cur_q):
            accepted_num += 1
        cur_q = nxt_q

        q_hist.append(nxt_q[0])
        #q_hist.append(nxt_q.tolist())
        if i % 50 == 0:
            print("progressed {}%".format(i * 100 / iters))
        if i % 1000 and adaptive:
            # every 1000 iterations, we re-estimate the covariance of estimated target
            cov = adaptive_metric(q_hist)
            inv_cov = np.linalg.inv(cov + np.identity(q_0.shape[0]) * 1e-8)

            def K(p, mass):
                res = np.zeros(p.shape[0])
                for i in range(p.shape[0]):
                    res[i] = np.dot(np.dot(p[i].T, inv_cov), p[i]) / 2

                return res

    print(accepted_num / iters)
    # corrplot(q_hist)
    return q_hist


def corrplot(trace, maxlags=100):
    plt.acorr(trace - np.mean(trace), normed=True, maxlags=maxlags)
    plt.xlim([0, maxlags])
    plt.show()


################# DO SOME SIMPLE TEST to compare various HMC #################
D = 1
mass = np.identity(D) * 1


def U(q):
    res = np.zeros(q.shape[0])
    for i in range(q.shape[0]):
        res[i] = np.dot(q[i] + 3, q[i] + 3) / 2

    return res


def K(p, mass):
    res = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        res[i] = np.dot(np.dot(p[i] + 2, np.identity(D)), p[i] + 2) / 4
    return res


# grad_U = lambda q:q+3
#
# #
# # #### visualize the bad proposal #####
x = np.linspace(-5, 5, 500)
# #
# #
# y = norm(-3,1).pdf(x)
#
# prop_y = norm(-2,.5**(1/2)).pdf(x)
# plt.plot(x, y,label = "Target")
# plt.plot(x,prop_y,label = "Proposed")
# plt.legend()
# plt.show()

#########################
# x0 = multivariate_normal(np.zeros(D),mass).rvs().reshape(D,)
# #q_hist = hmc(U,K,grad_U,5000,x0,leapfrog, adaptive=True)
# q_hist_adaptive = hmc(U, K, grad_U, 5000, x0, static_trajectory, adaptive=False)

# # #
# plt.figure(figsize=(8, 8))
# plt.subplot(2,1,1)
# plt.hist(q_hist, bins=50, density=True)
# plt.title("Simple HMC with bad kinetic")
# plt.plot(x, y,label = "Target")

# #
# plt.subplot(2,1,2)
# plt.hist(q_hist_adaptive, bins=50, density=True)
# plt.title("dynamic trajectory HMC")
# plt.plot(x, y,label = "Target")
# plt.plot(x,prop_y,label = "Prior")
# plt.legend()
# plt.show()


##################### Following are experiments on multi-modes #################################
def U(q):
    res = np.zeros(q.shape[0])
    for i in range(q.shape[0]):
        res[i] = - np.log(1 / 3 * np.exp(-(q[i] + 3) ** 2 / 2) + 2 / 3 * np.exp(-(q[i] - 1) ** 2 / (0.5) ** 2))

    return res


def K(p, mass):
    res = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        res[i] = np.dot(np.dot(p[i] + 1, np.identity(D)), p[i] + 1) / 4
    return res
grad_U = lambda q:-1/(U(q))*((1/3)*np.exp(-((q+3)**2)/2)*(-q-2) + (2/3)*np.exp(-((q-1)**2)/2)*(q+1))


y = lambda x:np.exp(-U(x))/1.4263607085121721
plt.plot(x, y(x),label = "Target")
plt.plot(x,norm(-1,1).pdf(x),label="Prior")
plt.legend()
plt.show()

x0 = multivariate_normal(np.zeros(D),mass).rvs().reshape(D,)
q_hist = hmc(U,K,grad_U,10000,x0,leapfrog, adaptive=True)
#q_hist_spiral = hmc(U,K,grad_U,5000,x0,simple_spiral_method, adaptive=False)

plt.figure(figsize=(8, 8))
plt.subplot(2,1,1)
plt.hist(q_hist[5000:], bins=50, density=True)
plt.title("Simple HMC on multi-mode")
plt.plot(x, y(x),label = "Target")
plt.legend()

# plt.subplot(2,1,2)
# plt.hist(q_hist_spiral, bins=50, density=True)
# plt.title("Spiral HMC on multi-mode")
# plt.plot(x, y(x),label = "Target")
# plt.legend()
plt.show()