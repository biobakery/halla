from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

'''Similarity wrapper functions (note: not distance!)

Given x, y, returns a tuple (similarity, p-value);
p-value will be None if not provided
'''
def nmi(x, y, return_pval=False):
    '''normalized mutual information, ranging from [0 .. 1]
    0: no mutual information; 1: perfect correlation
    '''
    if return_pval: return(normalized_mutual_info_score(x, y), None)
    return(normalized_mutual_info_score(x, y))

def pearson(x, y, return_pval=False):
    corr, pval = pearsonr(x, y)
    # TODO: enable tuning whether correlation should always be positive or not
    if return_pval: return(corr, pval)
    return(corr)

def spearman(x, y, return_pval=False):
    corr, pval = spearmanr(x, y)
    # TODO: enable tuning whether correlation should always be positive or not
    if return_pval: return(corr, pval)
    return(corr)

'''Constants
'''
SIM_FUNCS = {
    'nmi': nmi,
    'pearson': pearson,
    'spearman': spearman,
}

PVAL_PROVIDED = {
    'nmi': False,
    'pearson': True,
    'spearman': True,
}

def get_similarity_function(metric):
    '''Retrieve the right distance function
    according to the metric
    '''
    metric = metric.lower()
    if metric not in SIM_FUNCS:
        raise KeyError('The similarity metric is not available...')
    # only return the similarity scores
    return(SIM_FUNCS[metric])

def does_return_pval(metric):
    return(PVAL_PROVIDED[metric])

def similarity2distance(scores, metric):
    '''Convert similarity scores (numpy array) to distance given metric
    '''
    if type(scores) is not np.ndarray:
        raise ValueError('scores argument should be a numpy array!')
    metric = metric.lower()
    if metric == 'nmi': return(1 - scores)
    if metric == 'pearson' or metric == 'spearman':
        # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4498680/
        return(1 - scores)