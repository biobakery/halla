from .similarity import does_return_pval, get_similarity_function

import numpy as np
from scipy.stats import percentileofscore
import scipy.spatial.distance as spd
from statsmodels.stats.multitest import multipletests

def compute_permutation_test_pvalue(x, y, pdist_metric='nmi',
									permute_func='gpd', iters=10000, seed=None):
	'''Compute p-value using permutation test of the pairwise similarity between
		an X feature and a Y feature.
	'''
	def _compute_score(feat1, feat2):
		score = spd.cdist(feat1, feat2, metric=get_similarity_function(pdist_metric))
		return(score[0,0]) # original shape is (1,1)

	def _compute_pvalue(permuted_scores, gt_score, n):
		percentile = percentileofscore(permuted_scores, gt_score, kind='strict')
		pval = (((100 - percentile) / 100.0) * n + 1) / n # add one to prevent 0
		return(min(1.0, pval))

	if seed:
		np.random.seed(seed)
	if len(x.shape) == 1: x = x.reshape((1, len(x)))
	if len(y.shape) == 1: y = y.reshape((1, len(y)))
	permute_func = permute_func.lower()
	permuted_dist_scores = []
	# compute the ground truth scores for comparison later
	gt_score = _compute_score(x, y)
	permuted_y = np.copy(y[0])
	if permute_func == 'ecdf': # empirical cumulative dist. function
		for _ in range(iters):
			np.random.shuffle(permuted_y)
			# compute permuted score and append to the list
			permuted_dist_scores.append(_compute_score(x, permuted_y.reshape(y.shape)))
		# compute the significance
		pvalue = _compute_pvalue(np.array(permuted_dist_scores), gt_score, iters)
	return(pvalue)
				
def get_pvalue_table(X, Y, pdist_metric='nmi',
					 permute_func='gpd', permute_iters=1000, seed=None):
	'''Obtain pairwise p-value tables given features in X and Y
	'''
	# initiate table
	n, m = X.shape[0], Y.shape[0]
	pvalue_table = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			pvalue_table[i,j] = get_similarity_function(pdist_metric)(X[i,:], Y[j,:], return_pval=True)[1] \
				if does_return_pval(pdist_metric) else \
				compute_permutation_test_pvalue(
					X[i,:], Y[j,:], pdist_metric=pdist_metric,
					permute_func=permute_func, iters=permute_iters, seed=seed)
	return(pvalue_table)

def pvalues2qvalues(pvalues, alpha=0.05):
	'''Perform p-value correction for multiple tests (Benjamini/Hochberg)
	# TODO? add more methods
	Args:
	- pvalues: a 1D-array of pvalues
	- alpha  : family-wise error rate
	Return a tuple (adjusted p-value array, boolean array [True = reject]
	'''
	return(multipletests(pvalues, alpha=alpha, method='fdr_bh')[:2])
