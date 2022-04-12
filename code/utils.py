"""Helper functions for working with tensors."""

import matplotlib.pyplot as plt
import numpy as np
from random import uniform 
import random
from scipy.stats import rankdata
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def project_multipliers_to_L1_ball(multipliers, center, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors in the L1 ball with the given center multipliers and given radius.
    
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
        center: rank-1 `Tensor`, the Lagrange multipliers as the center.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto a L1 norm ball w.r.t. the Euclidean norm.
        The returned rank-1 `Tensor`  IS IN A SIMPLEX
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    assert radius >= 0
    # compute the offset from the center and the distance
    offset = tf.math.subtract(multipliers, center)
    dist = tf.math.abs(offset)
    # multipliers is not already a solution: optimum lies on the boundary (norm of dist == radius)
    # project *multipliers* on the simplex
    new_dist = project_multipliers_wrt_euclidean_norm(dist, radius=radius)
    signs = tf.math.sign(offset)
    new_offset =  tf.math.multiply(signs, new_dist)
    projection = tf.math.add(center, new_offset)
    projection = tf.maximum(0.0, projection)
    return projection

def project_multipliers_wrt_euclidean_norm(multipliers, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors with nonnegative elements that
    sum to at most "radius".

    From https://github.com/google-research/tensorflow_constrained_optimization/blob/master/tensorflow_constrained_optimization/python/train/lagrangian_optimizer.py
    
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto the
        feasible region w.r.t. the Euclidean norm.
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    if not multipliers.dtype.is_floating:
        raise TypeError("multipliers must have a floating-point dtype")
    multipliers_dims = multipliers.shape.dims
    if multipliers_dims is None:
        raise ValueError("multipliers must have a known rank")
    if len(multipliers_dims) != 1:
        raise ValueError("multipliers must be rank 1 (it is rank %d)" %
                     len(multipliers_dims))
    dimension = multipliers_dims[0].value
    if dimension is None:
        raise ValueError("multipliers must have a fully-known shape")

    def while_loop_condition(iteration, multipliers, inactive, old_inactive):
        """Returns false if the while loop should terminate."""
        del multipliers  # Needed by the body, but not the condition.
        not_done = (iteration < dimension)
        not_converged = tf.reduce_any(tf.not_equal(inactive, old_inactive))
        return tf.logical_and(not_done, not_converged)

    def while_loop_body(iteration, multipliers, inactive, old_inactive):
        """Performs one iteration of the projection."""
        del old_inactive  # Needed by the condition, but not the body.
        iteration += 1
        scale = tf.minimum(0.0, (radius - tf.reduce_sum(multipliers)) /
                           tf.maximum(1.0, tf.reduce_sum(inactive)))
        multipliers = multipliers + (scale * inactive)
        new_inactive = tf.cast(multipliers > 0, multipliers.dtype)
        multipliers = multipliers * new_inactive
        return (iteration, multipliers, new_inactive, inactive)

    iteration = tf.constant(0)
    inactive = tf.ones_like(multipliers, dtype=multipliers.dtype)

    # We actually want a do-while loop, so we explicitly call while_loop_body()
    # once before tf.while_loop().
    iteration, multipliers, inactive, old_inactive = while_loop_body(
      iteration, multipliers, inactive, inactive)
    iteration, multipliers, inactive, old_inactive = tf.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, multipliers, inactive, old_inactive),
      name="euclidean_projection")

    return multipliers



def generate_rand_vec_l1_ball(center, radius):
    '''
    generate a random vector IN THE SIMPLEX, and in the l1 ball given the center and radius
    '''
    n = len(center)
    splits = [0] + [uniform(0, 1) for _ in range(0,n-1)] + [1]
    splits.sort()
    diffs = [x - splits[i - 1] for i, x in enumerate(splits)][1:]
    diffs = map(lambda x:x*radius, diffs)
    diffs = np.array(list(diffs))
    signs = [(-1)**random.randint(0,1) for i in range(n)]
    diffs = np.multiply(diffs, signs)
    result = np.add(diffs, center)
    result = np.maximum(result, np.zeros(n))
    return result

######### Util functions for select best run ################

def find_best_candidate_index(objective_vector,
                              constraints_matrix,
                              rank_objectives=True,
                              max_constraints=True):
  """Heuristically finds the best candidate solution to a constrained problem.
  This function deals with the constrained problem:
  > minimize f(w)
  > s.t. g_i(w) <= 0 for all i in {0,1,...,m-1}
  Here, f(w) is the "objective function", and g_i(w) is the ith (of m)
  "constraint function". Given a set of n "candidate solutions"
  {w_0,w_1,...,w_{n-1}}, this function finds the "best" solution according
  to the following heuristic:
    1. If max_constraints=True, the m constraints are collapsed down to one
       constraint, for which the constraint violation is the maximum constraint
       violation over the m original constraints. Otherwise, we continue with m
       constraints.
    2. Across all models, the ith constraint violations (i.e. max{0, g_i(0)})
       are ranked, as are the objectives (if rank_objectives=True).
    3. Each model is then associated its MAXIMUM rank across all m constraints
       (and the objective, if rank_objectives=True).
    4. The model with the minimal maximum rank is then identified. Ties are
       broken using the objective function value.
    5. The index of this "best" model is returned.
  The "objective_vector" parameter should be a numpy array with shape (n,), for
  which objective_vector[i] = f(w_i). Likewise, "constraints_matrix" should be a
  numpy array with shape (n,m), for which constraints_matrix[i,j] = g_j(w_i).
  For more specifics, please refer to:
  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)
  This function implements the heuristic used for hyperparameter search in the
  experiments of Section 5.2.
  Args:
    objective_vector: numpy array of shape (n,), where n is the number of
      "candidate solutions". Contains the objective function values.
    constraints_matrix: numpy array of shape (m,n), where n is the number of
      "candidate solutions", and m is the number of constraints. Contains the
      constraint violation magnitudes.
    rank_objectives: bool, whether the objective function values should be
      included in the initial ranking step. If True, both the objective and
      constraints will be ranked. If False, only the constraints will be ranked.
      In either case, the objective function values will be used for
      tiebreaking.
    max_constraints: bool, whether we should collapse the m constraints down to
      one by maximizing over them, before searching for the best index.
  Returns:
    The index (in {0,1,...,n-1}) of the "best" model according to the above
    heuristic.
  Raises:
    ValueError: if "objective_vector" and "constraints_matrix" have inconsistent
      shapes.
  """
  nn, mm = np.shape(constraints_matrix)
  if (nn,) != np.shape(objective_vector):
    raise ValueError(
        "objective_vector must have shape (n,), and constraints_matrix (n, m),"
        " where n is the number of candidates, and m is the number of "
        "constraints")

  # If max_constraints is True, then we collapse the mm constraints down to one,
  # where this "one" is the maximum constraint violation across all mm
  # constraints.
  if mm > 1 and max_constraints:
    constraints_matrix = np.amax(constraints_matrix, axis=1, keepdims=True)
    mm = 1

  if rank_objectives:
    maximum_ranks = rankdata(objective_vector, method="min")
  else:
    maximum_ranks = np.zeros(nn, dtype=np.int64)
  for ii in xrange(mm):
    # Take the maximum of the constraint functions with zero, since we want to
    # rank the magnitude of constraint *violations*. If the constraint is
    # satisfied, then we don't care how much it's satisfied by (as a result, we
    # we expect all models satisfying a constraint to be tied at rank 1).
    ranks = rankdata(np.maximum(0.0, constraints_matrix[:, ii]), method="min")
    maximum_ranks = np.maximum(maximum_ranks, ranks)

  best_index = None
  best_rank = float("Inf")
  best_objective = float("Inf")
  for ii in xrange(nn):
    if maximum_ranks[ii] < best_rank:
      best_index = ii
      best_rank = maximum_ranks[ii]
      best_objective = objective_vector[ii]
    elif (maximum_ranks[ii] == best_rank) and (objective_vector[ii] <=
                                               best_objective):
      best_index = ii
      best_objective = objective_vector[ii]

  return best_index


######## Util functions for unaveraged results dicts. #########

# Adds best_idx results to results_dict.
def add_results_dict_best_idx_robust(results_dict, best_index):
    columns_to_add = ['train_ate_objective_vector', 'train_constraints_matrix']
    for column in columns_to_add:
        results_dict['best_' + column] = results_dict[column][best_index]
    return results_dict


# Adds best_idx results to results_dict.
def add_results_dict_best_idx_naive(results_dict, best_index):
    columns_to_add = ['train_ate_objective_vector']
    for column in columns_to_add:
        results_dict['best_' + column] = results_dict[column][best_index]
    return results_dict


######### Util functions for averaged results_dicts ################

# Outputs a results_dict with mean and standard dev for each metric for a list results_dicts.
def average_results_dict_fn(results_dicts):
    print('average_results_dict_fn')
    average_results_dict = {}
    for metric in results_dicts[0]:
        print('avg metric: ', metric)
        all_metric_arrays = []
        orig_shape = np.array(results_dicts[0][metric]).shape
        for results_dict in results_dicts: 
            all_metric_arrays.append(np.array(results_dict[metric]).flatten())
        all_metric_arrays = np.array(all_metric_arrays)
        mean_metric_flattened = np.mean(all_metric_arrays, axis=0)
        mean_metric = mean_metric_flattened.reshape(orig_shape)
        std_metric_flattened = np.std(all_metric_arrays, ddof=1, axis=0)
        std_metric = std_metric_flattened.reshape(orig_shape)
        average_results_dict[metric] = (mean_metric, std_metric)
    return average_results_dict


def print_avg_results_best_iter_robust(results_dict):
    def get_max_mean_std_best(mean_std_tuple):
        max_idx = np.argmax(mean_std_tuple[0])
        max_mean = mean_std_tuple[0][max_idx]
        max_std = mean_std_tuple[1][max_idx]
        return max_mean, max_std

    print("best iterate ATE objectives: ")
    print("%.4f \pm %.4f" % (float(results_dict['best_train_ate_objective_vector'][0]), float(results_dict['best_train_ate_objective_vector'][1])))
    
    print("best iterate max constraint violations: ")
    train_max_mean_std = get_max_mean_std_best(results_dict['best_train_constraints_matrix'])
   
    print("%.4f \pm %.4f" % (train_max_mean_std[0], train_max_mean_std[1]))    