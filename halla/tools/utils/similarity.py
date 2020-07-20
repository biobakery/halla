from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
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

def distcorr(x, y, return_pval=False):
    '''Perform distance correlation [0 .. 1]
    def src: https://en.wikipedia.org/wiki/Distance_correlation
    distance corr = 0 iif x and y are independent
    distance corr = 1 implies that dimensions of the linear subspaces spanned by x & y respectively are
      almost surely equal and if we assume that these subspaces are equal, then in this subspace
      y = A + bCx for some vector A, scalar b, and orthonormal matrix C
    code src: https://gist.github.com/satra/aa3d19a12b74e9ab7941 - much faster than the library dcor
    '''
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    # if 1D - add dummy axis
    if np.prod(x.shape) == len(x): x = x[:, None]
    if np.prod(y.shape) == len(y): y = y[:, None]
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError('Number of samples must match')
    a, b = squareform(pdist(x)), squareform(pdist(y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    
    if return_pval: return(dcor, None)
    return(dcor)

'''Constants
'''
SIM_FUNCS = {
    'nmi': nmi,
    'pearson': pearson,
    'spearman': spearman,
    'dcor': distcorr,
}

PVAL_PROVIDED = {
    'nmi': False,
    'pearson': True,
    'spearman': True,
    'dcor': False,
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
    if metric == 'nmi' or metric == 'dcor': return(1 - scores)
    if metric == 'pearson' or metric == 'spearman':
        # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4498680/
        return(1 - scores)