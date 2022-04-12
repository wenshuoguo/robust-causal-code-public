'''Helper functions used in the experiments with generating data/samples or calculate statistics/divergence'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import random

def tv_bound_mvn(mean_0, cov_0, mean_1, cov_1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one cov_0 and cov_1

    KL( (mean_0, cov_0) || (mean_1, cov_1))
         = .5 * ( tr(cov_1^{-1} cov_0) + log |cov_1|/|cov_0| + 
                  (mean_1 - mean_0)^T cov_1^{-1} (mean_1 - mean_0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = mean_0.shape[0]
    iS1 = np.linalg.inv(cov_1)
    diff = mean_1 - mean_0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ cov_0)
    det_term  = np.log(np.linalg.det(cov_1)/np.linalg.det(cov_0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(cov_1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
   
    KL = (.5 * (tr_term + det_term + quad_term - N))
    
    return (0.5 * KL)**0.5

def generate_tilde_X_samples(X_samples, Z_samples, z_list, mean_list, var_list, seed=88):
    '''
    generate noisy covariates
    
    z_list contains all values for Z
    
    mean_list and var_list contains the mean and variance of noise for each value of Z, in order
    
    return: tilde_X_samples
    
    '''
#     np.random.seed(seed=seed)
    n = X_samples.shape[0]
    m = X_samples.shape[1]
    tilde_X_samples = np.zeros((n,m))
    for count, z in enumerate(z_list):
        z_idxs = np.where(Z_samples == z)[0]  
        for z_idx in z_idxs: 
            tilde_x = np.array([i + np.random.normal(mean_list[count],var_list[count]**0.5) for i in X_samples[z_idx]])
            tilde_X_samples[z_idx] = tilde_x
    return tilde_X_samples

def compute_gamma_list_gaussian(X_mean, X_var, mean_list, var_list, d=2):
    '''
    compute a numpy array of gamma_z
    
    X_mean: scalar
    X_var: scalar
    mean_list: mean of noise for each z, (2,) for binary treatment
    var_list: variance of noise for each z, (2,) for binary treatment
    '''
    
    x_mean = np.full((d,), X_mean) # E[X], shape (d,)
        
    x_cov = np.zeros((d, d))
    np.fill_diagonal(x_cov, X_var)
        
    gamma_list = []
    for count, var in enumerate(var_list):
        noise_cov = np.zeros((d, d))
        np.fill_diagonal(noise_cov, var)
        
        noise_mean = np.full((d,), mean_list[count])
        #this only holds for gaussian distribution
        tilde_x_mean = x_mean + noise_mean
        tilde_x_cov = x_cov + noise_cov
        gamma_z = tv_bound_mvn(x_mean, x_cov, tilde_x_mean, tilde_x_cov)
        gamma_list.append(gamma_z)
    return gamma_list

def compute_ate_logistic(beta_0_hat=None, beta_1_hat=None, beta_2_hat=None, X_samples=None, X_weights=None):
    
    tau_hat = []
    for count, X_sample in enumerate(X_samples):
        tau_hat.append(expit(beta_0_hat + beta_1_hat + np.dot(beta_2_hat, X_sample))
                             -expit(beta_0_hat + np.dot(beta_2_hat, X_sample)))

    tau_hat = np.average(np.array(tau_hat), weights=X_weights)  
    return tau_hat

def fit_logistic(X_samples=None, Z_samples=None, Y_samples=None, X_weights=None, equal_weights=False):
    '''
    fit a logistic regression model between Y and Z, X:
    Y = expit(beta_0 + beta_1 * Z + <beta_2, X>)
    
    and compute the ATE estimate
    
    '''
    num_samples = X_samples.shape[0]
    if equal_weights==True:
#         print('all samples equally weighted')
        X_weights =  np.full((num_samples,), float(1/num_samples))  
    Z_X_samples = np.concatenate((Z_samples.reshape(-1,1), X_samples), axis=1)
    model = LogisticRegression(random_state=0, solver='lbfgs').fit(Z_X_samples, Y_samples, sample_weight=X_weights*num_samples)
    beta_0_hat = model.intercept_[0]
    beta_1_hat = model.coef_[0][0]
    beta_2_hat = model.coef_[0][1:]

    tau_hat = compute_ate_logistic(beta_0_hat=beta_0_hat, beta_1_hat=beta_1_hat, 
                                   beta_2_hat=beta_2_hat, X_samples=X_samples, X_weights=X_weights)
#     print('beta_0_hat, beta_1_hat, beta_2_hat: ', beta_0_hat, beta_1_hat, beta_2_hat)
    return tau_hat


def fit_logistic_frontdoor(X_samples=None, Z_samples=None, Y_samples=None, X_weights=None, equal_weights=False):
    '''
    fit a logistic regression model between Y and X:
    Y = expit(psi_0 + psi_1 * X)
    
    and compute the ATE estimate
    
    '''
    num_samples = X_samples.shape[0]
    if equal_weights==True:
#         print('all samples equally weighted')
        X_weights =  np.full((num_samples,), float(1/num_samples))  
#     Z_X_samples = np.concatenate((Z_samples.reshape(-1,1), X_samples), axis=1)
    model = LogisticRegression(random_state=0, solver='lbfgs').fit(X_samples.reshape(-1, 1), Y_samples, sample_weight=X_weights*num_samples)
    psi_0_hat = model.intercept_[0]
    psi_1_hat = model.coef_[0][0]

    tau_hat = compute_ate_logistic_frontdoor_binary(psi_0_hat=psi_0_hat, psi_1_hat=psi_1_hat, 
                                   X_samples=X_samples, Z_samples=Z_samples, X_weights=X_weights)
#     print('beta_0_hat, beta_1_hat, beta_2_hat: ', beta_0_hat, beta_1_hat, beta_2_hat)
    return tau_hat


def compute_ate_logistic_frontdoor_binary(psi_0_hat=None, psi_1_hat=None, X_samples=None, Z_samples=None, X_weights=None):
    
    x_probs = compute_binary_cond_probs(Z_samples=Z_samples, X_samples=X_samples)
    print('x_probs', x_probs)
    tau_hat = x_probs[2]* expit(psi_0_hat) + x_probs[3]* expit(psi_0_hat+psi_1_hat)-x_probs[0]* expit(psi_0_hat) - x_probs[1]* expit(psi_0_hat+psi_1_hat)

    return tau_hat

def compute_binary_cond_probs(Z_samples=None, X_samples=None):
    Z_indices = np.asarray([np.where(Z_samples == 0)[0] , np.where(Z_samples == 1)[0]])
    
    num_samples = X_samples.shape[0]
    x_counts = []
    for z_value, z_indice in enumerate(Z_indices):
      
        x_0_z = 0
        x_1_z = 0
        for idx in z_indice:
            if X_samples[idx]==0:
                x_0_z += 1
            else:
                x_1_z += 1
        x_counts.append(x_0_z)
        x_counts.append(x_1_z)
    #array of cond. probabilities [P(X=0|Z=0), P(X=1|Z=0),P(X=0|Z=1),P(X=1|Z=1)]
    x_probs = [x_counts[0]/len(Z_indices[0]),x_counts[1]/len(Z_indices[0]),x_counts[2]/len(Z_indices[1]),x_counts[3]/len(Z_indices[1])]
       
    return x_probs

def compute_gamma_frontdoor_binary(noise_level=None, X_samples=None, Z_samples=None, seed=998):
    np.random.seed(seed=seed)
    num_samples = X_samples.shape[0]
    #set noise parameters and compute TV 
    noise_idx = random.sample(range(num_samples), int(noise_level * num_samples))
    # print(noise_idx)
    tilde_X_samples = np.copy(X_samples)

    for i in noise_idx:
        if tilde_X_samples[i] == 0:
            tilde_X_samples[i] = 1

    #array of cond. probabilities [P(X=0|Z=0), P(X=1|Z=0),P(X=0|Z=1),P(X=1|Z=1)]
    x_probs = compute_binary_cond_probs(Z_samples=Z_samples, X_samples=X_samples)
#     print(x_probs)
    tilde_x_probs = compute_binary_cond_probs(Z_samples=Z_samples, X_samples=tilde_X_samples)
#     print(tilde_x_probs)
    gamma_list = [abs(x_probs[0] - tilde_x_probs[0]), abs(x_probs[1] - tilde_x_probs[1])]

#     print('gamma_list', gamma_list)
   
    return(gamma_list,tilde_X_samples)

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.

Implementation code is from: https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

