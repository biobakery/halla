from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
import math

SCIPY_AVAILABLE_METRICS = [
    'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean',
    'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
    'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
    'sqeuclidean', 'yule'
]

'''Distance functions

Given x, y, returns a tuple (distance, p-value);
p-value will be None if not provided
'''
def nmi(x, y, return_pval=False):
    if return_pval: return(normalized_mutual_info_score(x, y), None)
    return(normalized_mutual_info_score(x, y))

def pearson(x, y, return_pval=False):
    corr, pval = pearsonr(x, y)
    # TODO: enable tuning whether correlation should always be positive or not
    corr = 1 - abs(corr)
    if return_pval: return(corr, pval)
    return(corr)

def spearman(x, y, return_pval=False):
    corr, pval = spearmanr(x, y)
    # TODO: enable tuning whether correlation should always be positive or not
    corr = 1 - abs(corr)
    if return_pval: return(corr, pval)
    return(corr)

'''Constants
'''
DIST_FUNCS = {
    'nmi': nmi,
    'pearson': pearson,
    'spearman': spearman,
}

PVAL_PROVIDED = {
    'nmi': False,
    'pearson': True,
    'spearman': True,
}

def get_distance_function(metric):
    '''Retrieve the right distance function
    according to the metric
    '''
    metric = metric.lower()
    if metric in SCIPY_AVAILABLE_METRICS: return metric
    if metric not in DIST_FUNCS:
        raise KeyError('The pdist metric is not available...')
    # only return the distance scores
    return(DIST_FUNCS[metric])

def does_return_pval(metric):
    if metric in SCIPY_AVAILABLE_METRICS: return False
    return(PVAL_PROVIDED[metric])