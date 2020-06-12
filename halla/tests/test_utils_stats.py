import unittest
import sys
from os.path import dirname, abspath
import numpy as np
from scipy.stats import spearmanr, pearsonr
sys.path.append(dirname(dirname(abspath(__file__))))

from tools.utils import stats
from utils import compare_numpy_array

class TestStatsUtils(unittest.TestCase):
    '''Tests the p-value permutation test function
    '''
    def test_compute_permutation_test_pvalue_significant1(self):
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
    
    def test_compute_permutation_test_pvalue_significant2(self):
        np.random.seed(2)
        eps = 0.001
        # source: https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
        x = np.random.normal(size=1000) * 20
        y = x + np.random.normal(loc=2, scale=10, size=1000) * 10
        # get expected p-value
        _, expected_pvalue = pearsonr(x, y)
        test_pvalue = stats.compute_permutation_test_pvalue(x, y, pdist_metric='pearson',
                        permute_func='ecdf', iters=1000, seed=123)
        self.assertLessEqual(abs(test_pvalue - expected_pvalue), eps)
    
    def test_compute_permutation_test_pvalue_insignificant1(self):
        np.random.seed(2)
        eps = 0.05
        x = np.random.normal(size=50)
        y = np.random.normal(size=50)
        _, expected_pvalue = spearmanr(x, y)
        test_pvalue = stats.compute_permutation_test_pvalue(x, y, pdist_metric='spearman',
                        permute_func='ecdf', iters=1000, seed=123)
        self.assertLessEqual(abs(test_pvalue - expected_pvalue), eps)
    
    def test_compute_permutation_test_pvalue_insignificant2(self):
        np.random.seed(111)
        eps = 0.05
        x = np.random.normal(size=50)
        y = np.random.normal(size=50)
        _, expected_pvalue = pearsonr(x, y)
        test_pvalue = stats.compute_permutation_test_pvalue(x, y, pdist_metric='pearson',
                        permute_func='ecdf', iters=1000, seed=123)
        self.assertLessEqual(abs(test_pvalue - expected_pvalue), eps)