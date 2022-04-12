'''functions with training the DRO model'''

import tensorflow as tf
import time
import numpy as np
import random
import pandas as pd

import utils
import functions
import naive_training
import samplers

class DRO_Model(object):
    """Linear model with DRO constrained optimization.
    
     Args:
      feature_names: list of strings, a list of names of all feature columns.
      protected_columns: list of strings, a list of the names of all protected group columns 
        (column should contain values of 0 or 1 representing group membership).
      label_column: string, name of label column. Column should contain values of 0 or 1.
      maximum_lambda_radius: float, an optional upper bound to impose on the
        sum of the lambdas.
      maximum_p_radius: float, an optional upper bound to impose on the
        L1 norm of each row of phats - ptildes.
        equals to 2 * gamma
    
    Raises:
      ValueError: if "maximum_lambda_radius" is nonpositive.  
      ValueError: if "maximum_p_radius" is negative.  
    """
    def __init__(self, feature_names, Z_0_X_names, Z_1_X_names, protected_columns, label_column, 
                 Z_indices, maximum_lambda_radius=None, maximum_p_radius=[0.2, 0.2],bound_type='LB', y_type='binary', estimator_type='backdoor'):
        tf.reset_default_graph()
        tf.random.set_random_seed(123)
        
        self.feature_names = feature_names  
        self.Z_0_X_names = Z_0_X_names
        self.Z_1_X_names = Z_1_X_names
        
        self.protected_columns = protected_columns
        self.label_column = label_column
        self.z_column = ['Z_1']
        self.Z_indices = Z_indices
        self.num_samples = Z_indices[0].shape[0] + Z_indices[1].shape[0]
        self.Z_nums = [Z_indices[0].shape[0], Z_indices[1].shape[0]]
        self.bound_type = bound_type
        self.y_type = y_type
        self.estimator_type = estimator_type
       
        
        if (maximum_lambda_radius is not None and maximum_lambda_radius <= 0.0):
            raise ValueError("maximum_lambda_radius must be strictly positive")
        if (maximum_p_radius is not None and maximum_p_radius[0] < 0.0):
            raise ValueError("maximum_p_radius must be non negative")
        self._maximum_lambda_radius = maximum_lambda_radius
        maximum_p_radius = [min(1,maximum_p_radiu) for maximum_p_radiu in maximum_p_radius] 
        self._maximum_p_radius = maximum_p_radius
        
        # Set up feature and label tensors.
        num_features = len(self.feature_names)
        self.features_placeholder = tf.placeholder(tf.float32, shape=(None, num_features), name='features_placeholder')
     
        if self.estimator_type == 'ipw' or self.estimator_type == 'frontdoor':
            # for ipw, num_features is only x_names
            self.Z_0_X_placeholder = tf.placeholder(tf.float32, shape=(None, num_features+1), name='Z_0_X_placeholder')
            self.Z_1_X_placeholder = tf.placeholder(tf.float32, shape=(None, num_features+1), name='Z_1_X_placeholder')
        else:
            self.Z_0_X_placeholder = tf.placeholder(tf.float32, shape=(None, num_features), name='Z_0_X_placeholder')
            self.Z_1_X_placeholder = tf.placeholder(tf.float32, shape=(None, num_features), name='Z_1_X_placeholder')

        self.protected_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name=attribute+"_placeholder") for attribute in self.protected_columns]
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, 1), name='labels_placeholder')
        self.num_groups = len(self.protected_placeholders)
        if estimator_type=='ipw':
            self.z_placeholder = tf.placeholder(tf.float32, shape=(None, 1), name='z_placeholder')
            
            
        # model parameters
        if estimator_type == 'backdoor': 
            if self.y_type=='binary':
                self.w = tf.Variable(tf.random_normal(shape=[num_features, 1], stddev=1), name="weights")
                self.b = tf.Variable(tf.zeros([]), name="bias")
                self.theta_variables = [self.w, self.b]
                self.predictions_tensor = tf.matmul(self.features_placeholder , self.w) + self.b
            elif self.y_type == 'cont_MLE' or self.y_type == 'cont_MSE': 
                self.w = tf.Variable(tf.constant(value=0, shape=[num_features, 1], dtype=tf.float32), name="weights")
                self.sigma = tf.Variable(tf.ones([]), name="likelihood_std")
                self.theta_variables = [self.w, self.sigma]
                self.predictions_tensor = tf.matmul(self.features_placeholder , self.w)
        elif estimator_type == 'frontdoor': 
            if self.y_type=='cont_MLE':
                self.w = tf.Variable(tf.constant(value=0, shape=[num_features, 1], dtype=tf.float32), name="weights")
                self.b = tf.Variable(tf.zeros([]), name="bias")
                self.sigma = tf.Variable(tf.ones([]), name="likelihood_std")
                self.theta_variables = [self.w, self.b, self.sigma]
                self.predictions_tensor = tf.matmul(self.features_placeholder , self.w) + self.b
            elif self.y_type=='cont_MSE' or self.y_type=='binary':
                #for debug purposes.
#                 self.w = tf.Variable(tf.convert_to_tensor(np.reshape([0.38452649,0.40530883,0.5245951,0.48040632,0.67685436], (num_features, 1)), dtype=tf.float32), name="weights")
#                 print('self.w ', self.w )
#                 self.b = tf.Variable(tf.constant(value=0.19611572937598168, shape=[], dtype=tf.float32), name="bias")
                
                self.w = tf.Variable(tf.constant(value=0, shape=[num_features, 1], dtype=tf.float32), name="weights")
                self.b = tf.Variable(tf.zeros([]), name="bias")
#                 self.sigma = tf.Variable(tf.ones([]), name="likelihood_std")
                self.theta_variables = [self.w, self.b]
                self.predictions_tensor = tf.matmul(self.features_placeholder , self.w) + self.b
        elif estimator_type == 'ipw': 
            self.w = tf.Variable(tf.random_normal(shape=[num_features, 1], stddev=1), name="weights")
            self.b = tf.Variable(tf.zeros([]), name="bias")
            self.theta_variables = [self.w, self.b]
            self.predictions_tensor = tf.matmul(self.features_placeholder, self.w) + self.b

                            
    def feed_dict_helper(self, dataframe):
#         print('start feed_dict_helper')
#         print(dataframe[self.feature_names].shape, dataframe[self.Z_0_X_names].shape, dataframe[self.Z_1_X_names].shape)
        if self.estimator_type == 'ipw':
            feed_dict = {self.features_placeholder: dataframe[self.feature_names], 
                         self.Z_0_X_placeholder: dataframe[self.Z_0_X_names],
                         self.Z_1_X_placeholder: dataframe[self.Z_1_X_names],
                         self.labels_placeholder: dataframe[self.label_column],
                         self.z_placeholder: dataframe[self.z_column],}
        
        else:
            feed_dict = {self.features_placeholder: dataframe[self.feature_names], 
                         self.Z_0_X_placeholder: dataframe[self.Z_0_X_names],
                         self.Z_1_X_placeholder: dataframe[self.Z_1_X_names],
                         self.labels_placeholder: dataframe[self.label_column],}
        for i, protected_attribute in enumerate(self.protected_columns):
                feed_dict[self.protected_placeholders[i]] = dataframe[[protected_attribute]] 
#         print('finish feed_dict_helper')
        return feed_dict
    
    def project_lambdas(self, lambdas):
        """Projects the Lagrange multipliers onto the feasible region."""
        if self._maximum_lambda_radius:
            projected_lambdas = project_multipliers_wrt_euclidean_norm(
              lambdas, self._maximum_lambda_radius)
        else:
            projected_lambdas = tf.maximum(0.0, lambdas)
        return projected_lambdas
    
    def get_gradient_norm_constraints(self, constraints_slack=0.00):
#         with tf.GradientTape() as g: 
            if self.y_type == 'binary' and self.estimator_type == 'backdoor':
                neg_log_likelihoods = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions_tensor, labels=self.labels_placeholder)
                neg_log_likelihood_all_z = []
                for i, protected_attribute in enumerate(self.protected_columns):
                    neg_log_likelihoods_z = tf.reshape(tf.gather(neg_log_likelihoods, self.Z_indices[i]), (self.Z_nums[i],))
                    neg_log_likelihood_z = tf.divide(tf.reduce_sum(tf.multiply(neg_log_likelihoods_z, self.p_variables_list[i])),tf.reduce_sum(self.p_variables_list[i]))
                    neg_log_likelihood_all_z.append(neg_log_likelihood_z) 
                avg_neg_log_likelihood = 0
                for count, neg_log_likelihood_z in enumerate(neg_log_likelihood_all_z):
                    avg_neg_log_likelihood += neg_log_likelihood_z * (self.Z_indices[count].shape[0]/self.num_samples)
                avg_neg_log_likelihood = tf.convert_to_tensor(avg_neg_log_likelihood) 
                # Compute gradients
#                 gradients = tf.gradients(avg_neg_log_likelihood, self.theta_variables)
#                 constraints_list = [tf.norm(gradient) - constraints_slack for gradient in gradients]
                constraints_list = [avg_neg_log_likelihood - constraints_slack]
            
            elif self.y_type == 'binary' and self.estimator_type == 'frontdoor':
                neg_log_likelihoods = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions_tensor, labels=self.labels_placeholder)
                neg_log_likelihood_all_z = []
                for i, protected_attribute in enumerate(self.protected_columns):
                    neg_log_likelihoods_z = tf.reshape(tf.gather(neg_log_likelihoods, self.Z_indices[i]), (self.Z_nums[i],))
                    neg_log_likelihood_z = tf.divide(tf.reduce_sum(tf.multiply(neg_log_likelihoods_z, self.p_variables_list[i])),tf.reduce_sum(self.p_variables_list[i]))
                    neg_log_likelihood_all_z.append(neg_log_likelihood_z) 
                avg_neg_log_likelihood = 0
                for count, neg_log_likelihood_z in enumerate(neg_log_likelihood_all_z):
                    avg_neg_log_likelihood += neg_log_likelihood_z * (self.Z_indices[count].shape[0]/self.num_samples)
                avg_neg_log_likelihood = tf.convert_to_tensor(avg_neg_log_likelihood) 
                # Compute gradients
#                 gradients = tf.gradients(avg_neg_log_likelihood, self.theta_variables)
                constraints_list = [avg_neg_log_likelihood - constraints_slack]
                
            elif self.y_type == 'cont_MLE_norm' and self.estimator_type == 'backdoor':

                neg_log_likelihoods = tf.squared_difference(self.predictions_tensor, self.labels_placeholder)  #start with squared errors
                neg_log_likelihood_all_z = []
                for i, protected_attribute in enumerate(self.protected_columns):
                    neg_log_likelihoods_z = tf.reshape(tf.gather(neg_log_likelihoods, self.Z_indices[i]), (self.Z_nums[i],))
                    neg_log_likelihood_z = tf.divide(tf.reduce_sum(tf.multiply(neg_log_likelihoods_z, self.p_variables_list[i])),tf.reduce_sum(self.p_variables_list[i]))
                    neg_log_likelihood_all_z.append(neg_log_likelihood_z) 
                avg_neg_log_likelihood = 0
                for count, neg_log_likelihood_z in enumerate(neg_log_likelihood_all_z):
                    avg_neg_log_likelihood += neg_log_likelihood_z * (self.Z_indices[count].shape[0]/self.num_samples)
                MSE_weighted = tf.convert_to_tensor(avg_neg_log_likelihood) 
                avg_neg_log_likelihood = tf.log(self.sigma) + tf.multiply(0.5/self.sigma, MSE_weighted)
                # Compute gradients
                gradients = tf.gradients(avg_neg_log_likelihood, self.theta_variables)
                constraints_list = [tf.norm(gradient) - constraints_slack for gradient in gradients]
            
            elif self.y_type == 'cont_MSE' and self.estimator_type == 'backdoor':
                print('Calculate constraints for cont_MSE')

                neg_log_likelihoods = tf.squared_difference(self.predictions_tensor, self.labels_placeholder)  #start with squared errors
                self.MSE_unweighted_debug = tf.reduce_mean(neg_log_likelihoods)
                neg_log_likelihood_all_z = []
                for i, protected_attribute in enumerate(self.protected_columns):
                    neg_log_likelihoods_z = tf.reshape(tf.gather(neg_log_likelihoods, self.Z_indices[i]), (self.Z_nums[i],))
                    neg_log_likelihood_z = tf.divide(tf.reduce_sum(tf.multiply(neg_log_likelihoods_z, self.p_variables_list[i])),tf.reduce_sum(self.p_variables_list[i]))
                    neg_log_likelihood_all_z.append(neg_log_likelihood_z) 
                avg_neg_log_likelihood = 0
                for count, neg_log_likelihood_z in enumerate(neg_log_likelihood_all_z):
                    avg_neg_log_likelihood += neg_log_likelihood_z * (self.Z_indices[count].shape[0]/self.num_samples)
#                 if self.estimator_type == 'frontdoor':
#                     avg_neg_log_likelihood = avg_neg_log_likelihood*10
                MSE_weighted = tf.convert_to_tensor(avg_neg_log_likelihood) 
                self.MSE_weighted_debug=MSE_weighted
                print('MSE_weighted', MSE_weighted)
                constraints_list = [MSE_weighted - constraints_slack]
                
                print('constraints_list', constraints_list)
                
            elif self.y_type == 'cont_MSE' and self.estimator_type == 'frontdoor':
                print('Calculate constraints for cont_MSE')

                neg_log_likelihoods = tf.squared_difference(self.predictions_tensor, self.labels_placeholder)  #start with squared errors
#                 self.MSE_unweighted_debug = tf.reduce_mean(neg_log_likelihoods)
                neg_log_likelihood_all_z = []
                for i, protected_attribute in enumerate(self.protected_columns):
                    neg_log_likelihoods_z = tf.reshape(tf.gather(neg_log_likelihoods, self.Z_indices[i]), (self.Z_nums[i],))
                    neg_log_likelihood_z = tf.divide(tf.reduce_sum(tf.multiply(neg_log_likelihoods_z, self.p_variables_list[i])),tf.reduce_sum(self.p_variables_list[i]))
                    neg_log_likelihood_all_z.append(neg_log_likelihood_z) 
                avg_neg_log_likelihood = 0
                for count, neg_log_likelihood_z in enumerate(neg_log_likelihood_all_z):
                    avg_neg_log_likelihood += neg_log_likelihood_z * (self.Z_indices[count].shape[0]/self.num_samples)

                MSE_weighted = tf.convert_to_tensor(avg_neg_log_likelihood) 
#                 self.MSE_weighted_debug=MSE_weighted
#                 print('MSE_weighted', MSE_weighted)
                constraints_list = [MSE_weighted - constraints_slack]
               
                print('constraints_list', constraints_list)
            
            elif self.estimator_type == 'ipw':
                print('Calculate constraints for IPW estimator')

                neg_log_likelihoods = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions_tensor, labels=self.z_placeholder)
                neg_log_likelihood_all_z = []
                for i, protected_attribute in enumerate(self.protected_columns):
                    neg_log_likelihoods_z = tf.reshape(tf.gather(neg_log_likelihoods, self.Z_indices[i]), (self.Z_nums[i],))
                    neg_log_likelihood_z = tf.divide(tf.reduce_sum(tf.multiply(neg_log_likelihoods_z, self.p_variables_list[i])),tf.reduce_sum(self.p_variables_list[i]))
                    neg_log_likelihood_all_z.append(neg_log_likelihood_z) 
                avg_neg_log_likelihood = 0
                for count, neg_log_likelihood_z in enumerate(neg_log_likelihood_all_z):
                    avg_neg_log_likelihood += neg_log_likelihood_z * (self.Z_indices[count].shape[0]/self.num_samples)
                avg_neg_log_likelihood = tf.convert_to_tensor(avg_neg_log_likelihood) 
                
                constraints_list = [avg_neg_log_likelihood - constraints_slack]
           
           
            elif self.y_type == 'cont_MLE':
                #minimize the sigmoid of neg log likelihood function
                neg_log_likelihoods = tf.squared_difference(self.predictions_tensor, self.labels_placeholder)  #start with squared errors
                
                self.MSE_unweighted_debug = tf.reduce_mean(neg_log_likelihoods)
                
                neg_log_likelihood_all_z = []
                for i, protected_attribute in enumerate(self.protected_columns):
                 
                    neg_log_likelihoods_z = tf.reshape(tf.gather(neg_log_likelihoods, self.Z_indices[i]), (self.Z_nums[i],))
                    neg_log_likelihood_z = tf.divide(tf.reduce_sum(tf.multiply(neg_log_likelihoods_z, self.p_variables_list[i])),tf.reduce_sum(self.p_variables_list[i]))
                    neg_log_likelihood_all_z.append(neg_log_likelihood_z) 
                avg_neg_log_likelihood = 0
                for count, neg_log_likelihood_z in enumerate(neg_log_likelihood_all_z):
                    avg_neg_log_likelihood += neg_log_likelihood_z * (self.Z_indices[count].shape[0]/self.num_samples)
                MSE_weighted = tf.convert_to_tensor(avg_neg_log_likelihood) 
                self.MSE_weighted_debug=MSE_weighted
              
                avg_neg_log_likelihood = tf.log(self.sigma) + tf.multiply(tf.divide(0.5,(tf.multiply(self.sigma,self.sigma))), MSE_weighted)   
                
#                 #compute sigmoid
#                 avg_neg_log_likelihood = tf.divide(avg_neg_log_likelihood, tf.constant(10000, dtype=tf.float32))
#                 self.avg_neg_log_likelihood_debug =avg_neg_log_likelihood
#                 avg_neg_log_likelihood = tf.math.sigmoid(avg_neg_log_likelihood)
                
#                 self.avg_neg_log_likelihood_sig_debug =avg_neg_log_likelihood
#                 print('avg_neg_log_likelihood after sigmoid, ', avg_neg_log_likelihood)
              
                constraints_list = [avg_neg_log_likelihood - constraints_slack]
                print('constraints_list', constraints_list)
            
            return constraints_list

    def project_pbar(self, pbar_z, z, num_z):
        """Projects pbar_z onto the feasible region."""
    
        ptilde_z = np.full((num_z,), float(1/num_z), dtype=np.float32)
        ptilde_z = tf.convert_to_tensor(ptilde_z)
        projected_pbar_z = utils.project_multipliers_to_L1_ball(pbar_z, ptilde_z, self._maximum_p_radius[z])
        return projected_pbar_z
    
    def build_train_ops(self, learning_rate_theta=0.01, learning_rate_lambda=0.01, learning_rate_p_list=[0.01, 0.01], constraints_slack=1.0):
        """Builds operators that take gradient steps during training.
        
        Args: 
          learning_rate_theta: float, learning rate for theta parameter on descent step.
          learning_rate_lambda: float, learning rate for lambda parameter on ascent step.
          learning_rate_p_list: list of float, learning rate for ptilde parameters on ascent step.
          constraints_slack: float, amount of slack for constraints. New constraint will be
              original_constraint - constraints_slack
        
        """
        def make_projection_p(z, num_z):
            return lambda x: self.project_pbar(x, z, num_z)
        
        # Create variables p.
        self.p_variables_list = []
        for i in range(self.num_groups):
            initial_p = np.full((self.Z_nums[i],), float(1/self.Z_nums[i]), dtype=np.float32)
            self.p_variables_list.append(tf.compat.v2.Variable(
              initial_p,
              trainable=True,
              name="p_bar",
              dtype=tf.float32, 
              constraint=make_projection_p(i, self.Z_nums[i])))
 
        # Compute (ATE estimator) objective: tau hat 
        if self.estimator_type=='backdoor':
            if self.y_type == 'binary':
               
                tau_hat_list = tf.sigmoid(tf.matmul(self.Z_1_X_placeholder, self.w) + self.b) - tf.sigmoid(tf.matmul(self.Z_0_X_placeholder , self.w) + self.b)
               
            elif self.y_type == 'cont_MLE' or self.y_type == 'cont_MSE':
                tau_hat_list = tf.matmul(self.Z_1_X_placeholder , self.w) - tf.matmul(self.Z_0_X_placeholder , self.w)
            tau_hat_all_z = []
            self.sum_p_bar =[]
            for i, protected_attribute in enumerate(self.protected_columns):
                tau_hat_list_z = tf.reshape(tf.gather(tau_hat_list, self.Z_indices[i]),(self.Z_nums[i],))
                product = tf.multiply(tau_hat_list_z, self.p_variables_list[i])
                self.sum_p_bar.append(tf.reduce_sum(self.p_variables_list[i]))
                tau_hat_z = tf.divide(tf.reduce_sum(product),tf.reduce_sum(self.p_variables_list[i]))
                tau_hat_all_z.append(tau_hat_z) 
            tau_hat = 0
            for count, tau_hat_z in enumerate(tau_hat_all_z):
                tau_hat += tau_hat_z * (self.Z_nums[count]/self.num_samples)
            self.objective = tf.convert_to_tensor(tau_hat)
        elif self.estimator_type == 'frontdoor':
            if self.y_type == 'binary':
                tau_hat_list = tf.sigmoid(tf.matmul(self.features_placeholder , self.w) + self.b)
                tau_hat_all_z = []
                self.sum_p_bar =[]
                for i, protected_attribute in enumerate(self.protected_columns):
                    tau_hat_list_z = tf.reshape(tf.gather(tau_hat_list, self.Z_indices[i]),(self.Z_nums[i],))
                    product = tf.multiply(tau_hat_list_z, self.p_variables_list[i])
                    self.sum_p_bar.append(tf.reduce_sum(self.p_variables_list[i])) #for checkoing only.
                    tau_hat_z = tf.divide(tf.reduce_sum(product),tf.reduce_sum(self.p_variables_list[i]))
                    tau_hat_all_z.append(tau_hat_z) 
                tau_hat = tau_hat_all_z[1] - tau_hat_all_z[0]
                self.objective = tf.convert_to_tensor(tau_hat)
            elif self.y_type == 'cont_MSE' or self.y_type == 'cont_MLE':
                tau_hat_list = tf.matmul(self.features_placeholder , self.w) + self.b
                self.MSE_DRO = tf.reduce_mean(tf.squared_difference(tau_hat_list, self.labels_placeholder))
                
                tau_hat_all_z = []
#                 self.tau_hat_all_z_sum = []
                self.sum_p_bar =[] #for checking.
                
                for i, protected_attribute in enumerate(self.protected_columns):
                    tau_hat_list_z = tf.reshape(tf.gather(tau_hat_list, self.Z_indices[i]),(self.Z_nums[i],))
                    product = tf.multiply(tau_hat_list_z, self.p_variables_list[i])
                    self.sum_p_bar.append(tf.reduce_sum(self.p_variables_list[i])) #for checking only.
                    tau_hat_z = tf.divide(tf.reduce_sum(product),tf.reduce_sum(self.p_variables_list[i]))
                    tau_hat_all_z.append(tau_hat_z) 
                    
#                 self.tau_hat_all_z = tau_hat_all_z #for checking.
                tau_hat = tau_hat_all_z[1] - tau_hat_all_z[0]
                self.objective = tf.convert_to_tensor(tau_hat)
        elif self.estimator_type == 'ipw':
            prob_z_1 = self.Z_nums[1]/(np.sum(self.Z_nums))
            
            print('prob_z_1', prob_z_1)
            score_list = tf.sigmoid(self.predictions_tensor)
            score_list = tf.clip_by_value(score_list, clip_value_min=0.001, clip_value_max=0.999)
            self.score_list = score_list
            print('score_list',score_list)
            y_list_z_1 = tf.reshape(tf.gather(self.labels_placeholder, self.Z_indices[1]),(self.Z_nums[1],))
            self.y_list_z_1=y_list_z_1
            
            score_list_z_1 = tf.reshape(tf.gather(score_list, self.Z_indices[1]),(self.Z_nums[1],))
            self.score_list_z_1 = score_list_z_1
            
            term_1_list = tf.divide(y_list_z_1, score_list_z_1)
           
            term_2_list = tf.divide(y_list_z_1, tf.subtract(tf.ones_like(score_list_z_1),score_list_z_1))
            self.term_1_list = term_1_list
            self.term_2_list = term_2_list
            self.sum_p_bar =[]
            for i, protected_attribute in enumerate(self.protected_columns):
                self.sum_p_bar.append(tf.reduce_sum(self.p_variables_list[i])) #for checking only.
            
            tau_hat = tf.subtract(term_1_list, term_2_list)
            tau_hat = tf.divide(tf.reduce_sum(tf.multiply(tau_hat, self.p_variables_list[1])),tf.reduce_sum(self.p_variables_list[1]))
            self.objective = tf.multiply(tau_hat,prob_z_1)
                
  
        # Create lagrange multiplier variables lambda.
        constraints_list = self.get_gradient_norm_constraints(constraints_slack=constraints_slack)
        self.constraints = tf.convert_to_tensor(constraints_list)   
        self.num_constraints = len(constraints_list)
        print(' self.num_constraints',  self.num_constraints)
        initial_lambdas = np.ones((self.num_constraints,), dtype=np.float32)
        self.lambda_variables = tf.compat.v2.Variable(
          initial_lambdas,
          trainable=True,
          name="lambdas",
          dtype=tf.float32, 
          constraint=self.project_lambdas)   
        
        # Lagrangian loss to minimize
        if self.bound_type == 'LB':
            lagrangian_loss = self.objective + tf.tensordot(
              tf.cast(self.lambda_variables, dtype=self.constraints.dtype.base_dtype),
              self.constraints, 1)
        else:
            lagrangian_loss = self.objective - tf.tensordot(
              tf.cast(self.lambda_variables, dtype=self.constraints.dtype.base_dtype),
              self.constraints, 1)

        optimizer_theta = tf.train.AdamOptimizer(learning_rate_theta)
        optimizer_lambda = tf.train.AdamOptimizer(learning_rate_lambda)
        optimizer_p_list = []
        for i in range(len(learning_rate_p_list)):
            optimizer_p_list.append(tf.train.AdamOptimizer(learning_rate_p_list[i]))
        if self.bound_type == 'LB':
            self.train_op_theta = optimizer_theta.minimize(lagrangian_loss, var_list=self.theta_variables)
            self.train_op_lambda = optimizer_lambda.minimize(-lagrangian_loss, var_list=self.lambda_variables)
            self.train_op_p_list = []
            for i in range(self.num_groups):
                optimizer_p = optimizer_p_list[i]
                p_variable = self.p_variables_list[i]
                train_op_p = optimizer_p.minimize(lagrangian_loss, var_list=p_variable)
                self.train_op_p_list.append(train_op_p)
        else:
            self.train_op_theta = optimizer_theta.minimize(-lagrangian_loss, var_list=self.theta_variables)
            self.train_op_lambda = optimizer_lambda.minimize(lagrangian_loss, var_list=self.lambda_variables)
            self.train_op_p_list = []
            for i in range(self.num_groups):
                print('self.num_groups', self.num_groups)
                print('len(optimizer_p_list)', len(optimizer_p_list))
                optimizer_p = optimizer_p_list[i]
                p_variable = self.p_variables_list[i]
                train_op_p = optimizer_p.minimize(-lagrangian_loss, var_list=p_variable)
                self.train_op_p_list.append(train_op_p)
        return self.train_op_theta, self.train_op_lambda, self.train_op_p_list
    
    
def training_generator(model,
                       train_df,
                       minibatch_size=None,
                       num_iterations_per_loop=1,
                       num_loops=1):
    tf.set_random_seed(31337)
    num_rows = train_df.shape[0]
    p_variables_list_all_loop = []
    
    if minibatch_size is None:
        print('minibatch is off')
        minibatch_size = num_rows
    else:
        minibatch_size = min(minibatch_size, num_rows)
    permutation = list(range(train_df.shape[0]))
    random.shuffle(permutation)

    session = tf.Session()
    session.run((tf.global_variables_initializer(),
               tf.local_variables_initializer()))

    # Iterate through minibatches. Gradients are computed on each minibatch.
    minibatch_start_index = 0
    for n in range(num_loops):
        print('start loop ', n+1, 'in loops ', num_loops)
        loop_start_time = time.time()
        for _ in range(num_iterations_per_loop):
            minibatch_indices = []
            while len(minibatch_indices) < minibatch_size:
                minibatch_end_index = (
                minibatch_start_index + minibatch_size - len(minibatch_indices))
                if minibatch_end_index >= num_rows:
                    minibatch_indices += range(minibatch_start_index, num_rows)
                    minibatch_start_index = 0
                else:
                    minibatch_indices += range(minibatch_start_index, minibatch_end_index)
                    minibatch_start_index = minibatch_end_index
            minibatch_df = train_df.iloc[[permutation[ii] for ii in minibatch_indices]]
            # Update step on theta.
            session.run(
                  model.train_op_theta,
                  feed_dict=model.feed_dict_helper(minibatch_df))  

            # Update step on lambda.
            session.run(
                  model.train_op_lambda,
                  feed_dict=model.feed_dict_helper(minibatch_df))
            # Update step on p.
            for i in range(model.num_groups):
                session.run(
                      model.train_op_p_list[i],
                      feed_dict=model.feed_dict_helper(minibatch_df))
            
              
#         tau_hat_all_z_sum = session.run(model.tau_hat_all_z_sum, model.feed_dict_helper(train_df))
#         print('tau_hat_all_z_sum', tau_hat_all_z_sum)
        
        
#         tau_hat_all_z = session.run(model.tau_hat_all_z, model.feed_dict_helper(train_df))
#         print('tau_hat_all_z', tau_hat_all_z)
        
        objective = session.run(model.objective, model.feed_dict_helper(train_df))
        print('objective ', objective)  
                      
#         score_list = session.run(model.score_list, model.feed_dict_helper(train_df))
#         print('min/max score_list', np.min(score_list), np.max(score_list))
        
#         y_list_z_1 = session.run(model.y_list_z_1, model.feed_dict_helper(train_df))
#         print('min/max y_list_z_1', np.min(y_list_z_1), np.max(y_list_z_1))
        
#         score_list_z_1 = session.run(model.score_list, model.feed_dict_helper(train_df))
#         print('min/max score_list_z_1', np.min(score_list_z_1), np.max(score_list_z_1))
        
#         term_1_list = session.run(model.term_1_list, model.feed_dict_helper(train_df))
#         print('term_1_list', term_1_list)
        
#         term_2_list = session.run(model.term_2_list, model.feed_dict_helper(train_df))
#         print('term_2_list', term_2_list)


        sum_p_bar = session.run(model.sum_p_bar, model.feed_dict_helper(train_df))
        print('sum_p_bar',  sum_p_bar)
        
        constraints = session.run(model.constraints, model.feed_dict_helper(train_df))
        print('constraints, ', constraints)
        
#         theta_variables = session.run(model.theta_variables, model.feed_dict_helper(train_df))
#         print('theta_variables, ', theta_variables)
        
#         MSE_unweighted = session.run(model.MSE_unweighted_debug, model.feed_dict_helper(train_df))
#         print('MSE_unweighted, ', MSE_unweighted)
        
        
#         MSE_weighted = session.run(model.MSE_weighted_debug, model.feed_dict_helper(train_df))
#         print('MSE_weighted, ', MSE_weighted)
        
           
#         MSE_DRO = session.run(model.MSE_DRO, model.feed_dict_helper(train_df))
#         print('MSE_DRO, ', MSE_DRO)
        
        
#         ave_neg_log_likelihood = session.run(model.avg_neg_log_likelihood_debug, model.feed_dict_helper(train_df))
#         print('ave_neg_log_likelihood, ', ave_neg_log_likelihood)
        
#         avg_neg_log_likelihood_sig = session.run(model.avg_neg_log_likelihood_sig_debug, model.feed_dict_helper(train_df))
#         print('avg_neg_log_likelihood_sig, ', avg_neg_log_likelihood_sig)
        
        train_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(train_df))
#         print('predictions_tensor', train_predictions)
        
        lambda_variables = session.run(model.lambda_variables)
        p_variables_list = session.run(model.p_variables_list)
        print('finish loop ', n+1, 'in loops ', num_loops)
        print('time for this loop ',time.time() - loop_start_time)

        yield (objective, constraints, train_predictions, lambda_variables, p_variables_list)

def training_helper(model,
                    train_df,
                    minibatch_size = None,
                    num_iterations_per_loop=1,
                    num_loops=1):
    
    train_ate_objective_vector = []
    train_constraints_matrix = []  

    for objective, constraints, train_predictions, lambda_variables, p_variables_list in training_generator(
      model, train_df, minibatch_size, num_iterations_per_loop,
      num_loops):   
        train_ate_objective_vector.append(objective)
        train_constraints_matrix.append(constraints)
        train_df['predictions'] = train_predictions
    return {'train_ate_objective_vector': train_ate_objective_vector, 
            'train_constraints_matrix': train_constraints_matrix}


def get_results_for_learning_rates( X_samples=None, Z_samples=None,Y_samples=None, z_list=None, 
                                    mean_list=None, var_list=None, gamma_list=None, 
                                    constraints_slack=1, maximum_p_radius = [0.4, 0.4], 
                                    maximum_lambda_radius=1,
                                    learning_rates_theta = [0.01], #[0.001,0.01,0.1]
                                    learning_rates_lambda = [0.1], # [0.5, 1, 2]
                                    learning_rate_p_lists = [[0.001, 0.001]], 
                                    num_runs=1,  #10, num of splits
                                    minibatch_size=None,  #1000
                                    num_iterations_per_loop=2,  #100
                                    num_loops=1, bound_type='LB', seed=88): #30  
    ts = time.time()
     
    Z_0 = np.array([1-z for z in Z_samples])
    all_ones = np.array([1 for z in Z_samples])
    all_zeros = np.array([0 for z in Z_samples])
    dim = X_samples[0].shape[0]
    x_names = ["tilde_X_%d" % (i + 1) for i in range(dim)]
    feature_names = ['Z'] + x_names
    Z_0_X_names = ['all_zeros'] + x_names
    Z_1_X_names = ['all_ones'] + x_names
    protected_columns = ['Z_0', 'Z_1']
    label_column = ['Y']
    Z_indices = np.asarray([np.where(Z_samples == 0)[0] , np.where(Z_samples == 1)[0]])
    z_nums = [np.sum(Z_0), np.sum(Z_samples)]
    maximum_p_radius = [2*x for x in gamma_list]
    
    results_dicts_runs = []
    tau_hat_tilde_list = []
    best_bound_list = []
    best_constraints_list = []
    for i in range(num_runs):
        print('Split %d of %d' % (i+1, num_runs))
        t_split = time.time()
        
        #generate one set of \tilde X
        tilde_X_samples = functions.generate_tilde_X_samples(X_samples, Z_samples, z_list, mean_list, var_list, seed=seed+i)

        #prepare df 
        train_df = pd.DataFrame(np.concatenate((Z_samples.reshape(-1,1), 
                                      tilde_X_samples, 
                                      Y_samples.reshape(-1,1), 
                                      Z_0.reshape(-1,1), 
                                      Z_samples.reshape(-1,1), 
                                      all_zeros.reshape(-1,1), 
                                      all_ones.reshape(-1,1)), 
                                     axis=1), 
                      columns = feature_names + label_column + protected_columns + ['all_zeros', 'all_ones'])
        
        #use tilde X to compute ATE
        tau_hat_tilde= functions.fit_logistic(X_samples=tilde_X_samples, Z_samples=Z_samples, Y_samples=Y_samples, equal_weights=True)
       
        print('tau_hat_tilde', tau_hat_tilde)
        tau_hat_tilde_list.append(tau_hat_tilde)
        
        train_objectives = []
        train_constraints_matrix = []                 
        results_dicts = []
        learning_rates_iters_theta = []
        learning_rates_iters_lambda = []
        learning_rates_iters_p_list = []
                                   
        for learning_rate_p_list in learning_rate_p_lists:
            for learning_rate_theta in learning_rates_theta:
                for learning_rate_lambda in learning_rates_lambda:
                    t_start_iter = time.time() - ts
                    print("time since start:", t_start_iter)
                    print("begin optimizing learning rate p list:", learning_rate_p_list)
                    print("begin optimizing learning rate theta: %.3f learning rate lambda: %.3f" % (learning_rate_theta, learning_rate_lambda))
                   
                    
                    model = DRO_Model(feature_names, Z_0_X_names, Z_1_X_names, protected_columns, label_column, Z_indices,
                                      maximum_lambda_radius=maximum_lambda_radius, maximum_p_radius=maximum_p_radius, bound_type=bound_type)
                  
                    model.build_train_ops(learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda,learning_rate_p_list = learning_rate_p_list, constraints_slack=constraints_slack)
                    
                    # training_helper returns the list of errors and violations over each epoch. 
                    results_dict = training_helper(
                          model,
                          train_df,
                          minibatch_size=minibatch_size,
                          num_iterations_per_loop=num_iterations_per_loop,
                          num_loops=num_loops)                           
                                   
                    #find index for the best train iteration for this pair of hyper parameters
                    best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_ate_objective_vector']),np.array(results_dict['train_constraints_matrix']))
                    
                    train_objectives.append(results_dict['train_ate_objective_vector'][best_index_iters])
                    train_constraints_matrix.append(results_dict['train_constraints_matrix'][best_index_iters])
                                   
                    results_dict_best_idx = utils.add_results_dict_best_idx_robust(results_dict, best_index_iters)
                    results_dicts.append(results_dict_best_idx)
                    learning_rates_iters_theta.append(learning_rate_theta)
                    learning_rates_iters_lambda.append(learning_rate_lambda)
                    learning_rates_iters_p_list.append(learning_rate_p_list)
                    print("Finished learning rate p list", learning_rate_p_list)
                    print("Finished optimizing learning rate theta: %.3f learning rate lambda: %.3f " % (learning_rate_theta, learning_rate_lambda))
                    print("Time that this run took:", time.time() - t_start_iter - ts)
        
        #find the index of the best pair of hyper parameters        
        best_index = utils.find_best_candidate_index(np.array(train_objectives),np.array(train_constraints_matrix))
        best_results_dict = results_dicts[best_index]
        best_learning_rate_theta = learning_rates_iters_theta[best_index]
        best_learning_rate_lambda = learning_rates_iters_lambda[best_index]
        best_learning_rate_p_list = learning_rates_iters_p_list[best_index]
        print('best_learning_rate_theta,', best_learning_rate_theta)
        print('best_learning_rate_lambda', best_learning_rate_lambda)
        print('best_learning_rate_p_list', best_learning_rate_p_list)
        results_dicts_runs.append(best_results_dict)
        best_bound_list.append(best_results_dict['best_train_ate_objective_vector'])
        best_constraints_list.append(best_results_dict['best_train_constraints_matrix'])
        print("time it took for split", i+1, 'is', time.time() - t_split)
#     final_average_results_dict = utils.average_results_dict_fn(results_dicts_runs)
    
    return tau_hat_tilde_list, best_bound_list, best_constraints_list


def get_results_for_learning_rates_DRO_one_split(tilde_X_samples = None, X_samples=None, Z_samples=None,
                                             Y_samples=None, z_list=None, 
                                    mean_list=None, var_list=None, gamma_list=None, 
                                    constraints_slack=1, maximum_p_radius = [0.4, 0.4], 
                                    maximum_lambda_radius=1,
                                    learning_rates_theta = [0.01], #[0.001,0.01,0.1]
                                    learning_rates_lambda = [0.1], # [0.5, 1, 2]
                                    learning_rate_p_lists = [[0.001, 0.001]], 
                                    num_runs=1,  #10, num of splits
                                    minibatch_size=None,  #1000
                                    num_iterations_per_loop=2,  #100
                                    num_loops=1, bound_type='LB', y_type='cont_MSE', seed=88, estimator_type='backdoor'): #30  
    ts = time.time()
     
    Z_0 = np.array([1-z for z in Z_samples])
    all_ones = np.array([1 for z in Z_samples])
    all_zeros = np.array([0 for z in Z_samples])
    dim = X_samples[0].shape[0]
    x_names = ["tilde_X_%d" % (i + 1) for i in range(dim)]
    if estimator_type=='backdoor':
        feature_names = ['Z'] + x_names
    elif estimator_type == 'frontdoor' or estimator_type == 'ipw' :
        feature_names = x_names
    Z_0_X_names = ['all_zeros'] + x_names
    
    Z_1_X_names = ['all_ones'] + x_names
    
    protected_columns = ['Z_0', 'Z_1']
    label_column = ['Y']
    Z_indices = np.asarray([np.where(Z_samples == 0)[0] , np.where(Z_samples == 1)[0]])
#     print('Z_indices length', Z_indices[0].shape, Z_indices[1].shape)
    z_nums = [np.sum(Z_0), np.sum(Z_samples)]
    maximum_p_radius = [2*x for x in gamma_list]
    
    results_dicts_runs = []
    best_bound_list = []
    best_constraints_list = []
    for i in range(num_runs):
        print('Split %d of %d' % (i+1, num_runs))
        t_split = time.time()

        #prepare df 
        if estimator_type=='backdoor':
            train_df = pd.DataFrame(np.concatenate((Z_samples.reshape(-1,1), 
                                          tilde_X_samples, 
                                          Y_samples.reshape(-1,1), 
                                          Z_0.reshape(-1,1), 
                                          Z_samples.reshape(-1,1), 
                                          all_zeros.reshape(-1,1), 
                                          all_ones.reshape(-1,1)), 
                                         axis=1), 
                          columns = feature_names + label_column + protected_columns + ['all_zeros', 'all_ones'])
        elif (estimator_type=='frontdoor' and dim > 1) or estimator_type=='ipw':
            train_df = pd.DataFrame(np.concatenate((
                                          tilde_X_samples, 
                                          Y_samples.reshape(-1,1), 
                                          Z_0.reshape(-1,1), 
                                          Z_samples.reshape(-1,1), 
                                          all_zeros.reshape(-1,1), 
                                          all_ones.reshape(-1,1)), 
                                         axis=1), 
                          columns = feature_names + label_column + protected_columns + ['all_zeros', 'all_ones'])
        elif estimator_type=='frontdoor' and dim == 1:
            train_df = pd.DataFrame(np.concatenate((
                                          tilde_X_samples.reshape(-1,1), 
                                          Y_samples.reshape(-1,1), 
                                          Z_0.reshape(-1,1), 
                                          Z_samples.reshape(-1,1), 
                                          all_zeros.reshape(-1,1), 
                                          all_ones.reshape(-1,1)), 
                                         axis=1), 
                          columns = feature_names + label_column + protected_columns + ['all_zeros', 'all_ones'])
        
        train_objectives = []
        train_constraints_matrix = []                 
        results_dicts = []
        learning_rates_iters_theta = []
        learning_rates_iters_lambda = []
        learning_rates_iters_p_list = []
                                   
        for learning_rate_p_list in learning_rate_p_lists:
            for learning_rate_theta in learning_rates_theta:
                for learning_rate_lambda in learning_rates_lambda:
                    t_start_iter = time.time() - ts
                    print("time since start:", t_start_iter)
                    print("begin optimizing learning rate p list:", learning_rate_p_list)
                    print("begin optimizing learning rate theta: %.3f learning rate lambda: %.3f" % (learning_rate_theta, learning_rate_lambda))
                   
                    
                    model = DRO_Model(feature_names, Z_0_X_names, Z_1_X_names, protected_columns, label_column, Z_indices, maximum_lambda_radius=maximum_lambda_radius, maximum_p_radius=maximum_p_radius, bound_type=bound_type, y_type=y_type, estimator_type=estimator_type)
                  
                    model.build_train_ops(learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda,learning_rate_p_list = learning_rate_p_list, constraints_slack=constraints_slack)
                    
                    # training_helper returns the list of errors and violations over each epoch. 
                    results_dict = training_helper(
                          model,
                          train_df,
                          minibatch_size=minibatch_size,
                          num_iterations_per_loop=num_iterations_per_loop,
                          num_loops=num_loops)                           
                                   
                    #find index for the best train iteration for this pair of hyper parameters
                    if bound_type == 'UB' and estimator_type=='frontdoor':
                        results_dict_train_ate_objective_vector = np.array([-x for x in results_dict['train_ate_objective_vector']])
                        best_index_iters = utils.find_best_candidate_index(results_dict_train_ate_objective_vector,np.array(results_dict['train_constraints_matrix']))
                    else:
                        best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_ate_objective_vector']),np.array(results_dict['train_constraints_matrix']))
                    
                    train_objectives.append(results_dict['train_ate_objective_vector'][best_index_iters])
                    train_constraints_matrix.append(results_dict['train_constraints_matrix'][best_index_iters])
                                   
                    results_dict_best_idx = utils.add_results_dict_best_idx_robust(results_dict, best_index_iters)
                    results_dicts.append(results_dict_best_idx)
                    learning_rates_iters_theta.append(learning_rate_theta)
                    learning_rates_iters_lambda.append(learning_rate_lambda)
                    learning_rates_iters_p_list.append(learning_rate_p_list)
                    print("Finished learning rate p list", learning_rate_p_list)
                    print("Finished optimizing learning rate theta: %.3f learning rate lambda: %.3f " % (learning_rate_theta, learning_rate_lambda))
                    print("Time that this run took:", time.time() - t_start_iter - ts)
        
        #find the index of the best pair of hyper parameters        
        best_index = utils.find_best_candidate_index(np.array(train_objectives),np.array(train_constraints_matrix))
        best_results_dict = results_dicts[best_index]
        best_learning_rate_theta = learning_rates_iters_theta[best_index]
        best_learning_rate_lambda = learning_rates_iters_lambda[best_index]
        best_learning_rate_p_list = learning_rates_iters_p_list[best_index]
        print('best_learning_rate_theta,', best_learning_rate_theta)
        print('best_learning_rate_lambda', best_learning_rate_lambda)
        print('best_learning_rate_p_list', best_learning_rate_p_list)
        results_dicts_runs.append(best_results_dict)
        best_bound_list.append(best_results_dict['best_train_ate_objective_vector'])
        best_constraints_list.append(best_results_dict['best_train_constraints_matrix'])
        print("time it took for split", i+1, 'is', time.time() - t_split)
#     final_average_results_dict = utils.average_results_dict_fn(results_dicts_runs)
    
    return best_bound_list[0], best_constraints_list[0]
    

def compute_coverage_KS_backdoor(noise_mean=None, noise_var=None, 
                                      num_true_sets=None, num_samples=None,
                                      maximum_lambda_radius=None,
                                      ub_constraints_slack=0.01,
                                      ub_learning_rates_theta = [0.1], #[0.001,0.01,0.1]
                                        ub_learning_rates_lambda = [0.5], # [0.5, 1, 2]
                                        ub_learning_rate_p_lists = [[0.01, 0.01]], 
                                      lb_constraints_slack=0.01,
                                       lb_learning_rates_theta = [0.1], #[0.001,0.01,0.1]
                                        lb_learning_rates_lambda = [0.5], # [0.5, 1, 2]
                                        lb_learning_rate_p_lists = [[0.01, 0.01]],
                                        num_splits=20,  #num of X tilde sets sampled for each X
                                        minibatch_size=None,  #1000
                                        num_iterations_per_loop=30,  #100
                                        num_loops=20, seed=9876, tv_list=[0.1, 0.1], y_type='cont', norm_sample=False, estimator_type='backdoor', dataset = 'KS'):
     
    z_list = np.array([0,1])  #binary treatment
    #set noise parameters 
    mean_list = np.array([noise_mean,noise_mean])  #noise mean for each value of z
    var_list = np.array([noise_var, noise_var])  #noise var for each value of z
    true_tau_list = []
    LB_list = []
    UB_list = []
    tau_hat_mean_list = []
    tau_hat_std_list = []
    tau_hat_cover_list = []
    robust_cover_list = []
    
    for num_true_set in range(num_true_sets):
        t_start = time.time()
        print('start true set number ', num_true_set)
        if dataset == 'KS':
            X_samples, Y_samples, Z_samples = samplers.gen_KS_samples(num_samples, seed = seed+num_true_set, norm=norm_sample)   
        
        elif dataset == 'spam':
            X_samples, Y_samples, Z_samples = samplers.gen_ACIC_spam_samples(num_samples)   

        #use X to compute ATE
        print('Compute TRUE ATE')
        tau_hat_true = naive_training.get_results_for_learning_rates_Naive_one_split(tilde_X_samples=X_samples, Z_samples=Z_samples, Y_samples=Y_samples,z_list=z_list, 
                                                      constraints_slack=ub_constraints_slack,  
                                                        maximum_lambda_radius=maximum_lambda_radius,
                                                        learning_rates_theta = ub_learning_rates_theta, #[0.001,0.01,0.1]
                                                        learning_rates_lambda = ub_learning_rates_lambda, # [0.5, 1, 2]
                                                        minibatch_size=minibatch_size,  #1000
                                                        num_iterations_per_loop=num_iterations_per_loop,  #100
                                                        num_loops=num_loops, seed=seed, estimator=estimator_type)
        true_tau_list.append(tau_hat_true)
       
        local_robust_cover_list = []
        LB_split_list = []
        UB_split_list = []
        tau_hat_split_list = []
        for num_split in range(num_splits):
            print('Start split ', num_split)
            #generate one set of \tilde X
            
            tilde_X_samples = functions.generate_tilde_X_samples(X_samples, Z_samples, z_list, mean_list, var_list, seed=np.random.randint(-1000, 1000))

           #use tilde X to compute naive hat ATE
            print('Compute NAIVE hat ATE\n')
            tau_hat = naive_training.get_results_for_learning_rates_Naive_one_split(tilde_X_samples=tilde_X_samples, Z_samples=Z_samples, Y_samples=Y_samples,z_list=z_list,
                                          constraints_slack=ub_constraints_slack,  
                                            maximum_lambda_radius=maximum_lambda_radius,
                                            learning_rates_theta = ub_learning_rates_theta, #[0.001,0.01,0.1]
                                            learning_rates_lambda = ub_learning_rates_lambda, # [0.5, 1, 2]
                                            minibatch_size=minibatch_size,  #1000
                                            num_iterations_per_loop=num_iterations_per_loop,  #100
                                            num_loops=num_loops,  seed=seed,estimator=estimator_type)
            tau_hat_split_list.append(tau_hat)

            #compute TV bound (gamma_list)
            gamma_list=[]
            for count, z in enumerate(z_list):
                z_idxs = np.where(Z_samples == z)[0]  
                X_z = X_samples[z_idxs, :]
                tilde_X_z = tilde_X_samples[z_idxs, :]

#                 est_KL_z = functions.KLdivergence(X_z, tilde_X_z)
#                 print('est_KL_z', est_KL_z)
#                 gamma_z = (0.5*est_KL_z)**0.5
#                 print('estimated z, gamma_z', z, gamma_z)
                gamma_z = tv_list[count]
                gamma_list.append(gamma_z)

            #upper bound
            print('Compute Upper bound')
            UB, _ = get_results_for_learning_rates_DRO_one_split(tilde_X_samples=tilde_X_samples, X_samples=X_samples, 
                                                                             Z_samples=Z_samples, Y_samples=Y_samples,
                                                                          z_list=z_list, 
                                                                          mean_list=mean_list, var_list=var_list, 
                                                                            gamma_list=gamma_list, 
                                                                          constraints_slack=ub_constraints_slack,  
                                                                            maximum_lambda_radius=maximum_lambda_radius,
                                                                            learning_rates_theta = ub_learning_rates_theta, #[0.001,0.01,0.1]
                                                                            learning_rates_lambda = ub_learning_rates_lambda, # [0.5, 1, 2]
                                                                            learning_rate_p_lists = ub_learning_rate_p_lists, 
                                                                            minibatch_size=minibatch_size,  #1000
                                                                            num_iterations_per_loop=num_iterations_per_loop,  #100
                                                                            num_loops=num_loops, bound_type='UB', seed=seed, y_type=y_type, estimator_type=estimator_type)
            print('Computed Upper bound is: ', UB)
            #lower bound
            print('Compute Lower bound')
            LB, _ = get_results_for_learning_rates_DRO_one_split(tilde_X_samples=tilde_X_samples,X_samples=X_samples, Z_samples=Z_samples, Y_samples=Y_samples,
                                                          z_list=z_list, 
                                                          mean_list=mean_list, var_list=var_list, 
                                                            gamma_list=gamma_list, 
                                                          constraints_slack=lb_constraints_slack,  
                                                            maximum_lambda_radius=maximum_lambda_radius,
                                                            learning_rates_theta = lb_learning_rates_theta, #[0.001,0.01,0.1]
                                                            learning_rates_lambda = lb_learning_rates_lambda, # [0.5, 1, 2]
                                                            learning_rate_p_lists = lb_learning_rate_p_lists, 
                                                            minibatch_size=minibatch_size,  #1000
                                                            num_iterations_per_loop=num_iterations_per_loop,  #100
                                                            num_loops=num_loops, bound_type='LB', seed=seed, y_type=y_type,estimator_type=estimator_type)
 
            print('true ATE/ UB / LB/ Naive for this pair: ', tau_hat_true, UB, LB, tau_hat)
            LB_split_list.append(LB)
            UB_split_list.append(UB)
            if (UB >= tau_hat_true) and (LB <= tau_hat_true):
                local_robust_cover_list.append(1)
            else: 
                local_robust_cover_list.append(0)
        LB_list.append(LB_split_list)
        UB_list.append(UB_split_list)
        tau_hat_split_mean = np.mean(np.array(tau_hat_split_list))
        tau_hat_split_std = np.std(np.array(tau_hat_split_list))
        tau_hat_mean_list.append(tau_hat_split_mean)
        tau_hat_std_list.append(tau_hat_split_std)
        tau_hat_split_CI = tau_hat_split_std/(np.sqrt(num_splits)/3.92)
        
        if (tau_hat_split_mean + tau_hat_split_CI >= tau_hat_true) and (tau_hat_split_mean - tau_hat_split_CI <= tau_hat_true):
            tau_hat_cover_list.append(1)
        else: 
            tau_hat_cover_list.append(0)
        
        robust_cover_list = robust_cover_list + local_robust_cover_list
        print('Robust cover prob up to this set of X: number of set / Cover Prob.', num_true_set, np.mean(np.array(robust_cover_list)))
        print('Naive cover prob up to this set of X: number of set / Cover Prob.', num_true_set, np.mean(np.array(tau_hat_cover_list)))
        
        print('This set of X took: ', time.time()-t_start)
         
    tau_hat_cover_prob = np.mean(np.array(tau_hat_cover_list))
    tau_hat_cover_prob_std = np.std(np.array(tau_hat_cover_list))
    robust_cover_prob = np.mean(np.array(robust_cover_list))
    robust_cover_prob_std = np.std(np.array(robust_cover_list))
    print('true_tau_list', true_tau_list)
    print('tau_hat_mean_list', tau_hat_mean_list)
    print('tau_hat_std_list', tau_hat_std_list)
    print('UB_list', UB_list)
    print('LB_list', LB_list)
    
    return tau_hat_cover_prob, tau_hat_cover_prob_std, robust_cover_prob, robust_cover_prob_std




def get_results_for_learning_rates_frontdoor(X_samples=None, Z_samples=None,Y_samples=None, z_list=None, 
                                    noise_level=None,
                                    constraints_slack=1,
                                    maximum_lambda_radius=1,
                                    learning_rates_theta = [0.01], #[0.001,0.01,0.1]
                                    learning_rates_lambda = [0.1], # [0.5, 1, 2]
                                    learning_rate_p_lists = [[0.001, 0.001]], 
                                    num_runs=1,  #10, num of splits
                                    minibatch_size=None,  #1000
                                    num_iterations_per_loop=2,  #100
                                    num_loops=1, bound_type='LB', seed=88, estimator_type='frontdoor', y_type='binary'): #30  
    ts = time.time()
     
    Z_0 = np.array([1-z for z in Z_samples])
    all_ones = np.array([1 for z in Z_samples])
    all_zeros = np.array([0 for z in Z_samples])
    if y_type == 'binary':
        dim = 1
        x_names = ["tilde_X_1"]
        #x_names = ["tilde_X_%d" % (i + 1) for i in range(dim)]
    feature_names = x_names
    Z_0_X_names = ['all_zeros'] + x_names
    Z_1_X_names = ['all_ones'] + x_names
    protected_columns = ['Z_0', 'Z_1']
    label_column = ['Y']
    Z_indices = np.asarray([np.where(Z_samples == 0)[0] , np.where(Z_samples == 1)[0]])
    z_nums = [np.sum(Z_0), np.sum(Z_samples)]
#     maximum_p_radius = [2*x for x in gamma_list]
    
    results_dicts_runs = []
    tau_hat_tilde_list = []
    best_bound_list = []
    best_constraints_list = []
    for i in range(num_runs):
        print('Split %d of %d' % (i+1, num_runs))
        t_split = time.time()
        
        #generate one set of \tilde X
        gamma_list, tilde_X_samples = functions.compute_gamma_frontdoor_binary(noise_level=noise_level, X_samples=X_samples, Z_samples=Z_samples, seed=seed+i)
        print('gamma_list', gamma_list)
        maximum_p_radius = [gamma*2 for gamma in gamma_list]

        #prepare df 
        train_df = pd.DataFrame(np.concatenate((
                                      tilde_X_samples.reshape(-1,1), 
                                      Y_samples.reshape(-1,1), 
                                      Z_0.reshape(-1,1), 
                                      Z_samples.reshape(-1,1), 
                                      all_zeros.reshape(-1,1), 
                                      all_ones.reshape(-1,1)), 
                                     axis=1), 
                      columns = feature_names + label_column + protected_columns + ['all_zeros', 'all_ones'])
        
        #use tilde X to compute ATE
        tau_hat_tilde= functions.fit_logistic_frontdoor(X_samples=tilde_X_samples, Z_samples=Z_samples, Y_samples=Y_samples, equal_weights=True)
       
        print('tau_hat_tilde', tau_hat_tilde)
        tau_hat_tilde_list.append(tau_hat_tilde)
        
        train_objectives = []
        train_constraints_matrix = []                 
        results_dicts = []
        learning_rates_iters_theta = []
        learning_rates_iters_lambda = []
        learning_rates_iters_p_list = []
                                   
        for learning_rate_p_list in learning_rate_p_lists:
            for learning_rate_theta in learning_rates_theta:
                for learning_rate_lambda in learning_rates_lambda:
                    t_start_iter = time.time() - ts
                    print("time since start:", t_start_iter)
                    print("begin optimizing learning rate p list:", learning_rate_p_list)
                    print("begin optimizing learning rate theta: %.3f learning rate lambda: %.3f" % (learning_rate_theta, learning_rate_lambda))
                   
                    
                    model = DRO_Model(feature_names, Z_0_X_names, Z_1_X_names, protected_columns, label_column, Z_indices,maximum_lambda_radius=maximum_lambda_radius, bound_type=bound_type, estimator_type=estimator_type, y_type=y_type)
                  
                    model.build_train_ops(learning_rate_theta=learning_rate_theta, learning_rate_lambda=learning_rate_lambda,learning_rate_p_list = learning_rate_p_list, constraints_slack=constraints_slack)
                    
                    # training_helper returns the list of errors and violations over each epoch. 
                    results_dict = training_helper(
                          model,
                          train_df,
                          minibatch_size=minibatch_size,
                          num_iterations_per_loop=num_iterations_per_loop,
                          num_loops=num_loops)                           
                                   
                    #find index for the best train iteration for this pair of hyper parameters
                    best_index_iters = utils.find_best_candidate_index(np.array(results_dict['train_ate_objective_vector']),np.array(results_dict['train_constraints_matrix']))
                    
                    train_objectives.append(results_dict['train_ate_objective_vector'][best_index_iters])
                    train_constraints_matrix.append(results_dict['train_constraints_matrix'][best_index_iters])
                                   
                    results_dict_best_idx = utils.add_results_dict_best_idx_robust(results_dict, best_index_iters)
                    results_dicts.append(results_dict_best_idx)
                    learning_rates_iters_theta.append(learning_rate_theta)
                    learning_rates_iters_lambda.append(learning_rate_lambda)
                    learning_rates_iters_p_list.append(learning_rate_p_list)
                    print("Finished learning rate p list", learning_rate_p_list)
                    print("Finished optimizing learning rate theta: %.3f learning rate lambda: %.3f " % (learning_rate_theta, learning_rate_lambda))
                    print("Time that this run took:", time.time() - t_start_iter - ts)
        
        #find the index of the best pair of hyper parameters        
        best_index = utils.find_best_candidate_index(np.array(train_objectives),np.array(train_constraints_matrix))
        best_results_dict = results_dicts[best_index]
        best_learning_rate_theta = learning_rates_iters_theta[best_index]
        best_learning_rate_lambda = learning_rates_iters_lambda[best_index]
        best_learning_rate_p_list = learning_rates_iters_p_list[best_index]
        print('best_learning_rate_theta,', best_learning_rate_theta)
        print('best_learning_rate_lambda', best_learning_rate_lambda)
        print('best_learning_rate_p_list', best_learning_rate_p_list)
        results_dicts_runs.append(best_results_dict)
        best_bound_list.append(best_results_dict['best_train_ate_objective_vector'])
        best_constraints_list.append(best_results_dict['best_train_constraints_matrix'])
        print("time it took for split", i+1, 'is', time.time() - t_split)
#     final_average_results_dict = utils.average_results_dict_fn(results_dicts_runs)
    
    return tau_hat_tilde_list, best_bound_list, best_constraints_list



def compute_coverage_nonlinear_frontdoor(noise_mean=None, noise_var=None, num_true_sets=None, num_samples=None,
                                      maximum_lambda_radius=None,
                                      ub_constraints_slack=0.01,
                                      ub_learning_rates_theta = [0.1], #[0.001,0.01,0.1]
                                        ub_learning_rates_lambda = [0.5], # [0.5, 1, 2]
                                        ub_learning_rate_p_lists = [[0.01, 0.01]], 
                                      lb_constraints_slack=0.01,
                                       lb_learning_rates_theta = [0.1], #[0.001,0.01,0.1]
                                        lb_learning_rates_lambda = [0.5], # [0.5, 1, 2]
                                        lb_learning_rate_p_lists = [[0.01, 0.01]],
                                        num_splits=20,  #num of X tilde sets sampled for each X
                                        minibatch_size=None,  #1000
                                        num_iterations_per_loop=30,  #100
                                        num_loops=20, seed=3456, tv_list=[0.1, 0.1], y_type='cont_MSE', norm_sample=False, estimator_type='frontdoor', dataset=None):
     
    z_list = np.array([0,1])  #binary treatment
    #set noise parameters 
    mean_list = np.array([noise_mean,noise_mean])  #noise mean for each value of z
    var_list = np.array([noise_var, noise_var])  #noise var for each value of z
#     #set noise parameters 
#     noise_list = np.array([noise_level,noise_level])  #noise level for each value of z
# #     var_list = np.array([noise_var, noise_var])  #noise var for each value of z
    true_tau_list = []
    LB_list = []
    UB_list = []
    tau_hat_mean_list = []
    tau_hat_std_list = []
    tau_hat_cover_list = []
    robust_cover_list = []
    
    for num_true_set in range(num_true_sets):
        t_start = time.time()
        print('start true set number ', num_true_set)
        
        #generate true data
        if dataset == 'ihdp':
            X_samples, Y_samples,Z_samples = samplers.gen_IHDP_samples()
            
        else:
            X_samples, Y_samples, Z_samples = samplers.gen_nonlinear_frontdoor_samples(num_samples=num_samples, seed = seed+num_true_set, norm=norm_sample, fix_U=True)   

        print('Number of samples with Z=1, ', Z_samples.shape, np.sum(Z_samples))
        
        #use X to compute ATE
        print('Compute TRUE ATE')
        tau_hat_true = naive_training.get_results_for_learning_rates_Naive_one_split(tilde_X_samples=X_samples, Z_samples=Z_samples, Y_samples=Y_samples,z_list=z_list, constraints_slack=ub_constraints_slack,  
                                                        maximum_lambda_radius=maximum_lambda_radius,
                                                        learning_rates_theta = ub_learning_rates_theta, #[0.001,0.01,0.1]
                                                        learning_rates_lambda = ub_learning_rates_lambda, # [0.5, 1, 2]
                                                        minibatch_size=minibatch_size,  #1000
                                                        num_iterations_per_loop=num_iterations_per_loop,  #100
                                                        num_loops=num_loops, seed=seed, estimator = estimator_type)
        true_tau_list.append(tau_hat_true)
       
        local_robust_cover_list = []
        for num_split in range(num_splits):
            print('Start split ', num_split)
            
            #generate one set of \tilde X
            LB_split_list = []
            UB_split_list = []
            tau_hat_split_list = []
            
            tilde_X_samples = functions.generate_tilde_X_samples(X_samples, Z_samples, z_list, mean_list, var_list, seed=np.random.randint(-1000, 1000))

           #use tilde X to compute naive hat ATE
            print('Compute NAIVE hat ATE\n')
            tau_hat = naive_training.get_results_for_learning_rates_Naive_one_split(tilde_X_samples=tilde_X_samples, Z_samples=Z_samples, Y_samples=Y_samples,z_list=z_list, constraints_slack=ub_constraints_slack,  
                                            maximum_lambda_radius=maximum_lambda_radius,
                                            learning_rates_theta = ub_learning_rates_theta, #[0.001,0.01,0.1]
                                            learning_rates_lambda = ub_learning_rates_lambda, # [0.5, 1, 2]
                                            minibatch_size=minibatch_size,  #1000
                                            num_iterations_per_loop=num_iterations_per_loop,  #100
                                            num_loops=num_loops,  seed=seed,estimator = estimator_type)
            tau_hat_split_list.append(tau_hat)

            #set TV bound (gamma_list)
            gamma_list=[]
            for count, z in enumerate(z_list):
                z_idxs = np.where(Z_samples == z)[0]  
                X_z = X_samples[z_idxs, :]
                tilde_X_z = tilde_X_samples[z_idxs, :]

#                 est_KL_z = functions.KLdivergence(X_z, tilde_X_z)
#                 print('est_KL_z', est_KL_z)
#                 gamma_z = (0.5*est_KL_z)**0.5
#                 print('estimated z, gamma_z', z, gamma_z)
                gamma_z = tv_list[count]
                gamma_list.append(gamma_z)

            #upper bound
            print('Compute Upper bound')
            UB, _ = get_results_for_learning_rates_DRO_one_split(tilde_X_samples=tilde_X_samples, X_samples=X_samples, Z_samples=Z_samples, Y_samples=Y_samples,z_list=z_list, gamma_list=gamma_list, constraints_slack=ub_constraints_slack, maximum_lambda_radius=maximum_lambda_radius, learning_rates_theta = ub_learning_rates_theta, learning_rates_lambda=ub_learning_rates_lambda, learning_rate_p_lists = ub_learning_rate_p_lists, minibatch_size=minibatch_size, num_iterations_per_loop=num_iterations_per_loop,num_loops=num_loops, bound_type='UB', seed=seed, y_type=y_type, estimator_type = estimator_type)
            
            print('Computed Upper bound is: ', UB)
            #lower bound
            print('Compute Lower bound')
            LB, _ = get_results_for_learning_rates_DRO_one_split(tilde_X_samples=tilde_X_samples,X_samples=X_samples, Z_samples=Z_samples, Y_samples=Y_samples,
                                                          z_list=z_list, gamma_list=gamma_list, 
                                                          constraints_slack=lb_constraints_slack,  
                                                            maximum_lambda_radius=maximum_lambda_radius,
                                                            learning_rates_theta = lb_learning_rates_theta, #[0.001,0.01,0.1]
                                                            learning_rates_lambda = lb_learning_rates_lambda, # [0.5, 1, 2]
                                                            learning_rate_p_lists = lb_learning_rate_p_lists, 
                                                            minibatch_size=minibatch_size,  #1000
                                                            num_iterations_per_loop=num_iterations_per_loop,  #100
                                                            num_loops=num_loops, bound_type='LB', seed=seed, y_type=y_type, estimator_type = estimator_type)
 
            print('true ATE/ UB / LB/ Naive for this pair: ', tau_hat_true, UB, LB, tau_hat)
            LB_split_list.append(LB)
            UB_split_list.append(UB)
            if (UB >= tau_hat_true) and (LB <= tau_hat_true):
                local_robust_cover_list.append(1)
            else: 
                local_robust_cover_list.append(0)
        LB_list.append(LB_split_list)
        UB_list.append(UB_split_list)
        tau_hat_split_mean = np.mean(np.array(tau_hat_split_list))
        tau_hat_split_std = np.std(np.array(tau_hat_split_list))
        tau_hat_mean_list.append(tau_hat_split_mean)
        tau_hat_std_list.append(tau_hat_split_std)
        tau_hat_split_CI = tau_hat_split_std/(np.sqrt(num_splits)/3.92)
        
        if (tau_hat_split_mean + tau_hat_split_CI >= tau_hat_true) and (tau_hat_split_mean - tau_hat_split_CI <= tau_hat_true):
            tau_hat_cover_list.append(1)
        else: 
            tau_hat_cover_list.append(0)
        
        robust_cover_list = robust_cover_list + local_robust_cover_list
        print('Robust cover prob up to this set of X: number of set / Cover Prob.', num_true_set, np.mean(np.array(robust_cover_list)))
        print('Naive cover prob up to this set of X: number of set / Cover Prob.', num_true_set, np.mean(np.array(tau_hat_cover_list)))
        
        print('This set of X took: ', time.time()-t_start)
         
    tau_hat_cover_prob = np.mean(np.array(tau_hat_cover_list))
    tau_hat_cover_prob_std = np.std(np.array(tau_hat_cover_list))
    robust_cover_prob = np.mean(np.array(robust_cover_list))
    robust_cover_prob_std = np.std(np.array(robust_cover_list))
    print('true_tau_list', true_tau_list)
    print('tau_hat_mean_list', tau_hat_mean_list)
    print('tau_hat_std_list', tau_hat_std_list)
    print('UB_list', UB_list)
    print('LB_list', LB_list)
    
    return tau_hat_cover_prob, tau_hat_cover_prob_std, robust_cover_prob, robust_cover_prob_std