import unittest
import sys
from os.path import dirname, abspath
import numpy as np
import pandas as pd

from utils import compare_numpy_array

sys.path.append(dirname(dirname(abspath(__file__))))

from tools.utils.stats import stats

class TestStatsUtils(unittest.TestCase):

    '''Tests on computing permutation test p-values
    '''
    def test_compute_permutation_test_pvalue(self):
        
        