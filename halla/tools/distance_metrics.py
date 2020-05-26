from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr

SCIPY_AVAILABLE_METRICS = [
    'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean',
    'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
    'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
    'sqeuclidean', 'yule'
]

def pearson(x, y):
    #TODO: how to handle negative correlation?
    return(pearsonr(x, y)[0])

def spearman(x, y):
    #TODO: how to handle negative correlation?
    return(spearmanr(x, y)[0])

DIST_FUNCS = {
    'nmi': normalized_mutual_info_score,
    'pearson': pearson,
    'spearman': spearman,
}

def get_distance_function(metric):
    '''Retrieve the right distance function
    according to the metric
    '''
    metric = metric.lower()
    if metric in SCIPY_AVAILABLE_METRICS: return metric
    if metric not in DIST_FUNCS:
        raise KeyError('The pdist metric is not available...')
    return(DIST_FUNCS[metric])
