'''Naive model used in the experiments for continuous outcome'''

import tensorflow as tf
import time
import numpy as np
import random
import pandas as pd


import logging

import dowhy
from dowhy import CausalModel
import dowhy.datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import utils
import functions
import dro_training

def get_results_for_learning_rates_Naive_one_split(tilde_X_samples=None, Z_samples=None,
                                             Y_samples=None, z_list=None, 
                                    constraints_slack=1, maximum_p_radius = [0.4, 0.4], 
                                    maximum_lambda_radius=1,
                                    learning_rates_theta = [0.01], #[0.001,0.01,0.1]
                                    learning_rates_lambda = [0.1], # [0.5, 1, 2]
                                    num_runs=1,  #10, num of splits
                                    minibatch_size=None,  #1000
                                    num_iterations_per_loop=2,  #100
                                    num_loops=1, seed=88, estimator = 'backdoor'): #30  
    ts = time.time()
    num_samples = Z_samples.shape[0]
     
    Z_0 = np.array([1-z for z in Z_samples])
#     all_ones = np.array([1 for z in Z_samples])
#     all_zeros = np.array([0 for z in Z_samples])
#     print('debug', tilde_X_samples.shape)
    dim = tilde_X_samples[0].shape[0]
    
    x_names = ["tilde_X_%d" % (i + 1) for i in range(dim)]
    feature_names = ['Z'] + x_names
    Z_0_X_names = ['all_zeros'] + x_names
    Z_1_X_names = ['all_ones'] + x_names
    protected_columns = ['Z_0', 'Z_1']
    label_column = ['Y']
    Z_indices = np.asarray([np.where(Z_samples == 0)[0] , np.where(Z_samples == 1)[0]])
    z_nums = [np.sum(Z_0), np.sum(Z_samples)]

    #prepare df 
    train_df = pd.DataFrame(np.concatenate((Z_samples.reshape(-1,1), 
                                      tilde_X_samples, 
                                      Y_samples.reshape(-1,1), 
                                      Z_0.reshape(-1,1), 
                                      Z_samples.reshape(-1,1)), 
                                     axis=1), 
                      columns = feature_names + label_column + protected_columns)
    if estimator == 'backdoor':
        # Without graph
        model= CausalModel(
            data=train_df,
            treatment='Z',
            outcome='Y', common_causes=x_names,effect_modifiers=[],)


        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        causal_estimate_reg = model.estimate_effect(identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=True)
        print(causal_estimate_reg)
        ATE = causal_estimate_reg.value
    
    elif estimator =='ipw':
        train_df['Z'] = train_df['Z'].astype('bool')
        # Without graph
        model= CausalModel(
            data=train_df,
            treatment='Z',
            outcome='Y', common_causes=x_names,effect_modifiers=[],)

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        causal_estimate_ipw = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.propensity_score_weighting",
                                            target_units = "ate",
                                            method_params={"weighting_scheme":"ips_normalized_weight"})
        print(causal_estimate_ipw)
        ATE = causal_estimate_ipw.value
    
    elif estimator == 'frontdoor':
        reg = LinearRegression().fit(tilde_X_samples, Y_samples)
        beta_0 = reg.coef_
        beta_1 = reg.intercept_
#         print('reg.coef_, reg.intercept_: ', beta_0, beta_1)
        print('mean_squared_error: ', mean_squared_error(Y_samples, reg.predict(tilde_X_samples)))
        #compute ATE by frontdoor adj.
        ATE_1 = 0
        ATE_0 = 0
        for i, X_i in enumerate(tilde_X_samples):
            if Z_samples[i] == 1:
                ATE_1 += beta_1 + np.dot(beta_0, X_i)
            else:
                ATE_0 += beta_1 + np.dot(beta_0, X_i)
#         print('ATE_1 and ATE_0 without average, ', ATE_1, ATE_0)
        ATE_1 = ATE_1/np.sum(Z_samples)
        ATE_0 = ATE_0/(num_samples-np.sum(Z_samples))
#         print('ATE_1 and ATE_0 after average, ', ATE_1, ATE_0)
        ATE = ATE_1-ATE_0
    
    print("Causal Estimate is " + str(ATE))
    return ATE