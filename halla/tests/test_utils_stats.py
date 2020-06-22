import unittest
import sys
from os.path import dirname, abspath
import numpy as np
from scipy.stats import spearmanr, pearsonr
sys.path.append(dirname(dirname(abspath(__file__))))

from tools.utils import stats
from utils import compare_numpy_array

class TestStatsUtils(unittest.TestCase):
    '''Tests the p-value permutation test function; extreme cases
    '''
    def test_compute_permutation_test_pvalue_significant(self):
        np.random.seed(1)
        eps = 0.001
        # source: https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
        x = np.random.normal(size=1000) * 20
        y = x + np.random.normal(size=1000) * 10
        # get expected p-value
        _, expected_pvalue = spearmanr(x, y)
        test_pvalue = stats.compute_permutation_test_pvalue(x, y, pdist_metric='spearman',
                        permute_func='ecdf', iters=1000, seed=123)
        self.assertLessEqual(abs(test_pvalue - expected_pvalue), eps)
    
    def test_compute_permutation_test_pvalue_insignificant(self):
        np.random.seed(2)
        eps = 0.02
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        _, expected_pvalue  = pearsonr(x,y)
        test_pvalue = stats.compute_permutation_test_pvalue(x, y, pdist_metric='pearson',
                        permute_func='ecdf', iters=1000, seed=123)
        self.assertLessEqual(abs(test_pvalue - expected_pvalue), eps)
    
'''Other cases; programmatically create tests as attributes to TestStatsUtils
'''
def test_generator(mu_pair, var_pair, corr, sample_size):
    np.random.seed(123)
    eps = 0.05
    cov_matrix = np.array([
                    [var_pair[0], corr],
                    [corr, var_pair[1]]
                ])
    x, y = np.random.multivariate_normal(mu_pair, cov_matrix, sample_size).T
    def test(self):
        _, expected_pvalue = spearmanr(x, y)
        test_pvalue = stats.compute_permutation_test_pvalue(x, y, pdist_metric='spearman',
                        permute_func='ecdf', iters=1000, seed=123)
        self.assertLessEqual(abs(test_pvalue - expected_pvalue), eps)
    return test

corrs = [0.8, 0.03, -0.15, -0.4]
sample_sizes = [10, 50, 100]
mu_pairs = [[0, 0]]
variance_pairs = [[10.7, 20.3], [1, 50], [1,1]]

counter = 0
for corr in corrs:
    for sample_size in sample_sizes:
        for mu_pair in mu_pairs:
            for var_pair in variance_pairs:
                counter += 1
                test_name = 'test_compute_permutation_tet_pvalue_case%s' % counter
                test = test_generator(mu_pair, var_pair, corr, sample_size)
                setattr(TestStatsUtils, test_name, test)