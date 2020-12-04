from .similarity import does_return_pval, get_similarity_function, remove_missing_values

import numpy as np
import sys
import scipy.spatial.distance as spd
from statsmodels.stats.multitest import multipletests
from scipy.stats import genpareto
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from time import time

# retrieve package named 'eva' from R for GPD-related calculations
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
eva = importr('eva')

'''P-value computations by permutation test
'''
def compute_pvalue_ecdf(permuted_scores, gt_score, n):
    '''Compute the right-tailed test p-value using empirical cumulative distribution function given
    - permuted_scores: the scores computed by deriving association between x and shuffled y
    - gt_score       : the test statistic
    - n              : the number of iterations
    '''
    permuted_scores = np.array(permuted_scores)
    gt_score = np.abs(gt_score)
    pval = ((permuted_scores > gt_score).sum() + 1) / n
    return(min(1.0, pval))

def compute_pvalue_gpd(permuted_scores, gt_score, n):
    '''Approximate the p-value using generalized pareto distribution given
    Code converted to python from R: https://github.com/goncalves-lab/waddR/blob/master/R/PValues.R
    - permuted_scores: the scores computed by deriving association between x and shuffled y
    - gt_score       : the test statistic
    - n              : the number of iterations
    '''

    def get_pvalue(sorted_scores, stat, n):
        # approximate the gpd tail
        n_exceed = 250
        is_gpd_fitted = False
        while n_exceed >= 10:
            exceedances = sorted_scores[:n_exceed]
            # check if the n_exceed largest permutation values follow GPD
            #   with Anderson-Darling goodness-of-fit test
            try:
                ad = eva.gpdAd(FloatVector(exceedances))
                ad_pval = ad.rx2('p.value')[0]
            except:
                n_exceed -= 10
                continue
            # H0 = exceedances come from a GPD
            if ad_pval > 0.05:
                is_gpd_fitted = True
                break
            n_exceed -= 10
        if not is_gpd_fitted:
            print('GPD good fit is never reached - use ECDF instead...')
            return(None)
        # compute the exceedance threshold t
        t = float((sorted_scores[n_exceed] + sorted_scores[n_exceed-1])/2)
        # estimate shape and scale params with maximum likelihood
        gpd_fit = eva.gpdFit(FloatVector(sorted_scores), threshold=t, method='mle')
        scale, shape = gpd_fit.rx2('par.ests')[0], gpd_fit.rx2('par.ests')[1]

        # compute GPD p-value
        f_gpd = genpareto.cdf(x=gt_score-t, c=shape, scale=scale)
        return(n_exceed / n * (1 - f_gpd))

    gt_score = abs(gt_score)
    # approximate the tail
    sorted_scores = sorted(permuted_scores, reverse=True)
    return(get_pvalue(sorted_scores, gt_score, n))

def compute_permutation_test_pvalue(x, y, pdist_metric='nmi', permute_func='gpd',
                                    iters=10000, speedup=True, alpha=0.05, seed=None):
    '''Compute two-sided p-value using permutation test of the pairwise similarity between
        an X feature and a Y feature by:
        - set all scores and test statistic to be absolute values
        - do right-tailed test
    '''
    def _compute_score(feat1, feat2):
        score = spd.cdist(feat1, feat2, metric=get_similarity_function(pdist_metric))
        return(np.abs(score[0,0])) # original shape is (1,1)

    if seed:
        np.random.seed(seed)
    if len(x.shape) == 1: x = x.reshape((1, len(x)))
    if len(y.shape) == 1: y = y.reshape((1, len(y)))
    permute_func = permute_func.lower()
    permuted_dist_scores = []
    # compute the ground truth scores for comparison later
    rmx, rmy = remove_missing_values(x,y)
    if (np.unique(rmx).shape[0] == 1 or np.unique(rmy).shape[0] == 1):
        return(1)
    gt_score = _compute_score(x, y)
    permuted_y = np.copy(y[0])
    best_pvalue = 1.0
    permutation_num = iters
    for iter in range(iters):
        np.random.shuffle(permuted_y)
        # compute permuted score and append to the list
        permuted_dist_scores.append(_compute_score(x, permuted_y.reshape(y.shape)))
        if (iter+1) % 100 == 0 and speedup:
            curr_pvalue = compute_pvalue_ecdf(permuted_dist_scores, gt_score, iter+1)
            if curr_pvalue <= best_pvalue:
                best_pvalue = curr_pvalue
            elif curr_pvalue > alpha and (iter+1) >= 300:
                # only break if curr_pvalue > best_pvalue, curr_pvalue > alpha, iters >= 300 (arbitrary)
                permutation_num = iter + 1
                break
    if permute_func == 'ecdf': # empirical cumulative dist. function
        return(compute_pvalue_ecdf(permuted_dist_scores, gt_score, permutation_num))
    # gpd algorithm - Knijnenburg2009, Ge2012
    # compute M - # null samples exceeding the test statistic
    # recall that gt_score is positive
    M = len([1 for score in permuted_dist_scores if score > gt_score])
    # if M >= 10, use ecdf
    if M >= 10:
        return(compute_pvalue_ecdf(permuted_dist_scores, gt_score, permutation_num))
    
    # attempt to use gpd
    pval = compute_pvalue_gpd(permuted_dist_scores, gt_score, permutation_num)
    if pval is None:
        return(compute_pvalue_ecdf(permuted_dist_scores, gt_score, permutation_num))
    return(pval)

def get_pvalue_table(X, Y, pdist_metric='nmi', permute_func='gpd', permute_iters=1000,
                     permute_speedup=True, alpha=0.05, seed=None):
    '''Obtain pairwise p-value tables given features in X and Y
    '''
    # initiate table
    n, m = X.shape[0], Y.shape[0]
    pvalue_table = np.zeros((n, m))
    if does_return_pval(pdist_metric):
        for i in tqdm(range(n)):
            for j in range(m):
                pvalue_table[i,j] = get_similarity_function(pdist_metric)(X[i,:], Y[j,:], return_pval=True)[1]
    else:
        with Pool() as pool:
            pvalue_table = pool.starmap(compute_permutation_test_pvalue, [(X[i,:], Y[j,:], pdist_metric,
                                                                            permute_func, permute_iters,
                                                                            permute_speedup, alpha, seed)\
                                                                            for i in range(n) for j in range(m)])
            pvalue_table = np.array(pvalue_table).reshape((n, m))
    return(pvalue_table)

def test_pvalue_run_time(X, Y, pdist_metric='nmi', permute_func='gpd', permute_iters=1000,
                     permute_speedup=True, alpha=0.05, seed=None):
    '''
    Run a p-value computation test and return the time it took and a message extrapolating to the full dataset.
    '''
    test_start = time()
    
    if does_return_pval(pdist_metric):
        get_similarity_function(pdist_metric)(X[1,:], Y[1,:], return_pval=True)[1]
    else:
        compute_permutation_test_pvalue(X[1,:], Y[1,:],
                                    pdist_metric=pdist_metric, 
                                    permute_func=permute_func,
                                    iters=permute_iters,
                                    speedup=permute_speedup, alpha=alpha, seed=seed)    
    
    test_end = time()
    test_length = test_end - test_start
    extrapolated_time = test_length * X.shape[0] * Y.shape[0]
    timing_string = "The first p-value computation took about " + str(round(test_length, 2)) + " seconds. Extrapolating from this, computing the entire p-value table should take around " + str(round(extrapolated_time,2)) + " seconds..."
    return(extrapolated_time, timing_string)

def pvalues2qvalues(pvalues, method='fdr_bh', alpha=0.05):
    '''Perform p-value correction for multiple tests (Benjamini/Hochberg)
    Args:
    - pvalues: a 1D-array of pvalues
    - alpha  : family-wise error rate
    Return a tuple (adjusted p-value array, boolean array [True = reject]
    '''
    return(multipletests(pvalues, alpha=alpha, method=method)[:2])

def compute_result_power(significant_blocks, true_assoc):
    '''Compute power (recall: TP / condition positive) given args:
    - significant blocks: a list of significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - true_assoc        : A matrix with row ~ X features and col ~ Y features containing
                          1 if association exists or 0 if not
    '''
    positive_cond = np.sum(true_assoc).astype(int)
    positive_true = 0
    for block in significant_blocks:
        for i,j in itertools.product(block[0], block[1]):
            if true_assoc[i][j] == 1: positive_true += 1
    return(positive_true * 1.0 / positive_cond)

def compute_result_fdr(significant_blocks, true_assoc):
    '''Compute fdr (FP / predicted condition positive) given args:
    - significant blocks: a list of significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - true_assoc        : A matrix with row ~ X features and col ~ Y features containing
                          1 if association exists or 0 if not
    '''
    false_positive = 0
    predicted_positive = 0
    for block in significant_blocks:
        for i,j in itertools.product(block[0], block[1]):
            predicted_positive += 1
            if true_assoc[i][j] != 1: false_positive += 1
    if predicted_positive == 0: return(np.nan)
    return(false_positive * 1.0 / predicted_positive) 