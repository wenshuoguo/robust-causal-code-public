'''Samplers used in the experiments'''

import numpy as np
import random
from scipy.special import expit
import math
import pandas as pd


def gen_IHDP_samples():
    
    a = 10
    c = 1
    var_x = 2
    
    df_u = pd.read_csv('ihdp_data/x.csv', sep=",")
    df_u = df_u.to_numpy()
    U_samples = df_u[:,1:] 
    U_samples = (U_samples - U_samples.mean(axis=0)) / U_samples.std(axis=0)
    df_z = pd.read_csv('ihdp_data/z.csv', sep=",")
    df_z = df_z.to_numpy()
    Z_samples = df_z[:,1:] 
    
    X_samples = []
    Y_samples = []
    dim = U_samples[0].shape[0]
    for i, u_i in enumerate(U_samples):
        x_i = np.random.normal(c*Z_samples[i], var_x**0.5)
        X_samples.append(x_i)
        b = np.random.choice(np.arange(0,5), p=[0.5, 0.2, 0.15, 0.1, 0.05], size = (dim,))
        y_i = Z_samples[i] * x_i + np.dot(b,u_i)
        Y_samples.append(y_i)
    X_samples = np.array(X_samples)
    Y_samples = np.array(Y_samples)
    
    return(X_samples, Y_samples, Z_samples)


def gen_ACIC_spam_samples(num_samples):
    '''
    generate a dataset using the ACIC spam dataset
    '''
    df = pd.read_csv('spam_binMod11.csv', sep=",")
    df = df[:num_samples]
    
    all_data = df.to_numpy()
    
    X_samples = all_data[:,2:] 
    Y_samples = all_data[:,0]
    Z_samples = all_data[:,1]
    
    return(X_samples, Y_samples, Z_samples)
    

def gen_KS_samples(num_samples, seed=9876, norm=False):
    '''
    simulate a dataset according to Kang and Schafer:
    binary Z, continuous Y
    
    Xi1 = exp(Ui1/2), Xi2 = Ui2/{1+exp(Ui1)} + 10, Xi3 = (Ui1Ui3 + 0.6)^3, and Xi4 =
    (Ui2 +Ui4 +20)^2
    
    '''
    np.random.seed(seed=seed)
    
    U_mean = np.ones((4,))
    U_cov = np.identity(4)
    U_samples = np.random.multivariate_normal(U_mean, U_cov, size=num_samples)
    
    X_samples = []
    Z_samples = []
    Y_samples = []
    if norm == False:
        for u in U_samples:
            x_1 = math.exp(u[0]/2)
            x_2 = u[1]/(1+math.exp(u[0])) + 10
            x_3 = (u[0] * u[2]+0.6)**3
            x_4 = (u[1] + u[3]+20)**2
            X_samples.append([x_1, x_2, x_3, x_4]) 
            y = 210+27.4*u[0] +13.72*u[1] + 13.7*u[2] + 13.7*u[3] + np.random.normal(0,1)
            Y_samples.append(y)
            p = min(1,math.exp(-u[0] -2*u[1] -0.25*u[2] -0.1*u[3]))
            z = np.random.binomial(1, p)
            Z_samples.append(z)
    if norm == True:
        for u in U_samples:
            x_1 = math.exp(u[0]/2)
            x_2 = u[1]/(1+math.exp(u[0])) + 10
            x_3 = (u[0] * u[2]+0.6)**3
            x_4 = (u[1] + u[3]+20)**2
            X_samples.append([x_1, x_2, x_3, x_4]) 
            y = 210+27.4*u[0] +13.72*u[1] + 13.7*u[2] + 13.7*u[3] + np.random.normal(0,1)
            Y_samples.append(y)
            p = min(1,math.exp(-u[0] -2*u[1] -0.25*u[2] -0.1*u[3]))
            z = np.random.binomial(1, p)
            Z_samples.append(z)
            
        Y_samples = np.array(Y_samples)
        X_samples = np.array(X_samples)
        Y_samples = (Y_samples - Y_samples.mean(axis=0)) / Y_samples.std(axis=0)
        X_samples = (X_samples - X_samples.mean(axis=0)) / X_samples.std(axis=0)
    
    return(np.array(X_samples), np.array(Y_samples), np.array(Z_samples))
    

class simple_logistic_sampler:
    """Samples data from a simple logistic model."""
    X_strata = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
    
    def __init__(self, x_mean, x_var, a1_true, beta_0, beta_1, beta_2, d=2):
        """Initialize true model parameters.
        
        The true data generating model is:
        P(Z = 1 | X = x) = expit(a0_true + <a1_true, x>)
        
        P(Y = 1 | X = x, Z = z) = expit(beta_0 + beta_1 * z + <beta_2, x>)
        
        P(Y(0) = 1 | X = x) = expit(beta_0 + <beta_2, x>)
        P(Y(1) = 1 | X = x) = expit(beta_0 + beta_1 + <beta_2, x>)
        
        x_mean: scalar
        x_var: scalar, variance
        """ 
        self.a1_true = a1_true
        self.d = d
        self.X_mean = np.full((d,), x_mean) # E[X], shape (d,)
        
        self.X_cov = np.zeros((d, d))
        np.fill_diagonal(self.X_cov, x_var)
        
        self.a0_true = -np.dot(self.a1_true, self.X_mean)  #this makes P(Z=1) = 0.5
        
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def _get_X_samples(self, num_samples, dist_type='gaussian', seed=888):
        """Returns a numpy array with shape (num_samples, d) where each row represents a sample of X. """
        np.random.seed(seed=seed)
        if dist_type == 'bernoulli':
            '''
            for bernoulli, we take self.X_mean[0] which is the scalar p parameter
            '''
            X_samples = np.random.binomial(1, self.X_mean[0], (num_samples, self.d))
        elif dist_type == 'gaussian':
            X_samples = np.random.multivariate_normal(self.X_mean, self.X_cov, num_samples)
        return X_samples 
        
    def prob_Z_given_X(self, x):
        """ Returns P(Z = 1 | X = x), or the true propensity score.
        P(Z = 1 | X = x) = expit(a0_true + <a1_true, x>)
        """
        return expit(self.a0_true + np.dot(self.a1_true, x))
    
    def prob_Y_given_X_Z(self, x, z):
        if z == 0 or z == 1:
            return expit(self.beta_0 + self.beta_1 * z + np.dot(self.beta_2, x))
        else:
            raise("z must be binary.")
    
    def get_samples(self, num_samples, dist_type='gaussian', seed=888):
        """Returns numpy arrays Z_samples, X_samples, Y_samples."""

        X_samples = self._get_X_samples(num_samples, dist_type=dist_type, seed=seed)
        Z_samples = []
        Y_samples = []
      
        for x in X_samples:
            z = np.random.binomial(1, self.prob_Z_given_X(x))
            Z_samples.append(z)
            y = np.random.binomial(1, self.prob_Y_given_X_Z(x, z))
            Y_samples.append(y)
            
        return np.array(Z_samples), X_samples, np.array(Y_samples)
    

class simple_logistic_sampler_frontdoor:
    """Samples data from a simple logistic model for Frontdoor adjustment."""
   
    def __init__(self, u_mean, u_var, a1_true, beta_0, beta_1, beta_2, gamma_0, gamma_1, d=2):
        """Initialize true model parameters.
        
        U is generated from a multivariate Gaussian.
        
        The true data generating model is:
        P(Z = 1 | U = u) = expit(a0_true + <a1_true, u>)
        
        P(Y = 1 | X = x, U = u) = expit(beta_0 + beta_1 * x + <beta_2, u>)
        
        P(X = 1 | Z = z) = expit(gamma_0 + gamma_1 * z)
     
        
        u_mean: scalar, for u_i
        u_var: scalar, variance, for u_i
        """ 
        
        self.a1_true = a1_true
        self.d = d
        self.U_mean = np.full((d,), u_mean) # E[X], shape (d,)
        
        self.U_cov = np.zeros((d, d))
        np.fill_diagonal(self.U_cov, u_var)
        
        self.a0_true = -np.dot(self.a1_true, self.U_mean)  #this makes P(Z=1) = 0.5
        
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        
    def _get_U_samples(self, num_samples, dist_type='gaussian', seed=888):
        """Returns a numpy array with shape (num_samples, d) where each row represents a sample of U. """
        np.random.seed(seed=seed)
        if dist_type == 'bernoulli':
            '''
            for bernoulli, we take self.X_mean[0] which is the scalar p parameter
            '''
            U_samples = np.random.binomial(1, self.U_mean[0], (num_samples, self.d))
        elif dist_type == 'gaussian':
            U_samples = np.random.multivariate_normal(self.U_mean, self.U_cov, num_samples)
        return U_samples 
        
    def prob_Z_given_U(self, u):
        """ Returns P(Z = 1 | U = u), or the true propensity score.
        P(Z = 1 | U = u) = expit(a0_true + <a1_true, u>)
        """
        return expit(self.a0_true + np.dot(self.a1_true, u))
    
    def prob_X_given_Z(self, z):
        """ Returns P(X = 1 | Z = z), or the true propensity score.
        P(X = 1 | Z = z) = expit(gamma_0 + gamma_1 * z)
        """
        return expit(self.gamma_0 + self.gamma_1 * z)
    
    def prob_Y_given_X_U(self, x, u):
        if x == 0 or x == 1:
            return expit(self.beta_0 + self.beta_1 * x + np.dot(self.beta_2, u))
        else:
            raise("z must be binary.")
    
    def get_samples(self, num_samples, dist_type='gaussian', seed=888):
        """Returns numpy arrays Z_samples, X_samples, Y_samples."""

        U_samples = self._get_U_samples(num_samples, dist_type=dist_type, seed=seed)
        X_samples = []
        Z_samples = []
        Y_samples = []
      
        for u in U_samples:
            z = np.random.binomial(1, self.prob_Z_given_U(u))
            Z_samples.append(z)
            x = np.random.binomial(1, self.prob_X_given_Z(z))
            X_samples.append(x)
            y = np.random.binomial(1, self.prob_Y_given_X_U(x, u))
            Y_samples.append(y)
            
        return np.array(Z_samples), np.array(X_samples), np.array(Y_samples)
    
def gen_nonlinear_frontdoor_samples(num_samples=None, seed=5678, d=5, norm=False, fix_U=False):
    """ Generate a dataset using a nolinear model with continuous outcome for Frontdoor adjustment.
   
        Generation follows from this paper: Estimating Causal Effects Using Weighting-Based Estimators, Jung at al, at https://causalai.net/r54.pdf 
        
        U is generated from a Gaussian N(-2, 1)
        
        The true data generating model is:
        P(Z = 1 | U = u) = expit(U + epsilon_z), epsilon_z ~ Gaussian(0,0.5)
        
        X \in R^d, for each dimension:
        P(X = 1 | Z = z) = expit(c_1 + c_2 * z + epsilon_x), epsilon_x ~ Gaussian(-1,1), c_1, c_2 
        from a Gaussian(-2, 1)
        
        P(Y = 1 | X = x, U = u) = expit(2* <beta, x> + u + epsilon_y),  epsilon_x ~ Gaussian(0,1), 
        beta ~ Gaussian(1,1)
        
        input: d, positive integer, dimension of X
        
        """ 
    np.random.seed(seed=seed)
    
    if fix_U==False:
        U_samples = np.random.normal(-2, 1, size=num_samples)
    elif fix_U==True:
        U_samples=np.array([np.random.normal(-2, 1)]*num_samples)
        
    c_1 = np.random.normal(0, 1)
    c_2 = np.random.normal(1, 1)
    beta = np.random.normal(1, 1, size=d)
    
      
    X_samples = []
    Z_samples = []
    Y_samples = []
    if norm == False:
        for u in U_samples:
          
            p_z = expit(u + np.random.normal(0, 0.5))
            z = np.random.binomial(1, p_z)
            Z_samples.append(z)
            
            x = []
            for j in range(d):
                x_j = np.random.binomial(1, expit(c_1 + c_2 * z + np.random.normal(-1, 1)))
                x.append(x_j)
            X_samples.append(x)
            
            y = expit(2* np.dot(beta, np.array(x)) + u + np.random.normal(0, 1))
            Y_samples.append(y)
   
    return(np.array(X_samples), np.array(Y_samples), np.array(Z_samples))
    
    
    