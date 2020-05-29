from .distance import does_return_pval, get_distance_function

import numpy as np
from scipy.stats import percentileofscore
import scipy.spatial.distance as spd

def compute_permutation_test_pvalue(x, y, pdist_metric='nmi', pdist_args=None,
									permute_func='gpd', iters=1000, seed=None):
	def _compute_pvalue(n, permute_scores, compared_score):
		percentile = percentileofscore(permute_scores, compared_score, kind='strict')
		pval = (((100 - percentile) / 100.0) * n + 1) / n # add one to prevent 0
		return(min(1.0, pval))

	if seed:
		np.random.seed(seed)
	if len(x.shape) == 1: x = x.reshape((1, len(x)))
	if len(y.shape) == 1: y = y.reshape((1, len(y)))
	permute_func = permute_func.lower()
	permuted_dist_scores = []
	compared_score = spd.cdist(x, y, metric=get_distance_function(pdist_metric), **pdist_args) \
		if pdist_args else spd.cdist(x, y, metric=get_distance_function(pdist_metric))
	compared_score = compared_score[0,0]
	pvalue = 1.0
	if permute_func == 'ecdf': # empirical cumulative dist. function
		for iter_i in range(iters):
			permuted_y = np.random.permutation(y[0]).reshape((y.shape))
			permuted_score = spd.cdist(x, permuted_y, metric=get_distance_function(pdist_metric), **pdist_args) \
				if pdist_args else spd.cdist(x, permuted_y, metric=get_distance_function(pdist_metric))
			# permuted_score's shape is (1, 1)
			permuted_dist_scores.append(permuted_score[0,0])
			if (iter_i+1) % 100 == 0:
				curr_pvalue = _compute_pvalue(iter_i+1, permuted_dist_scores, compared_score)
				if curr_pvalue > pvalue: break
				pvalue = curr_pvalue
	return(pvalue)
				
def get_pvalue_table(X, Y, pdist_metric='nmi', pdist_args=None,
					 permute_func='gpd', permute_iters=1000, seed=None):
	'''Obtain pairwise p-value tables given features in X and Y
	'''
	# initiate table
	n, m = X.shape[0], Y.shape[0]
	pvalue_table = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			pvalue_table[i,j] = get_distance_function(pdist_metric)(X[i,:], Y[j,:], return_pval=True)[1] \
				if does_return_pval(pdist_metric) else \
				compute_permutation_test_pvalue(
					X[i,:], Y[j,:], pdist_metric=pdist_metric, pdist_args=pdist_args,
					permute_func=permute_func, iters=permute_iters, seed=seed)
	return(pvalue_table)