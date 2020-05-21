# TODO: implement metric functions
SCIPY_AVAILABLE_METRICS = [
    'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean',
    'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
    'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
    'sqeuclidean', 'yule'
]

def nmi():
   print('nmi') 

# def pearson():

# def spearman():

DIST_FUNCS = {
    'nmi': nmi,
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
