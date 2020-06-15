import unittest
import sys
from os.path import dirname, abspath
import numpy as np
sys.path.append(dirname(dirname(abspath(__file__))))

from tools.utils import tree
from utils import compare_numpy_array

class TestTreeUtils(unittest.TestCase):
    '''Test is_densely_associated function
    '''
    def test_is_densely_associated_wrongtype1(self):
        # has to be a numpy array
        self.assertRaises(ValueError, tree.is_densely_associated, [True, False])
    
    def test_is_densely_associated_wrongtype2(self):
        # has to be boolean, not int
        self.assertRaises(ValueError, tree.is_densely_associated, np.array([0, 1, 0]))
    
    def test_is_densely_associated_result1(self):
        block = np.array([False] * 10)
        # should be true if the threshold is 1
        self.assertTrue(tree.is_densely_associated(block, fnr_thresh=1))
    
    def test_is_densely_associated_result2(self):
        block = np.array([ # True proportion = 0.4
            False, False, True, True, False,
            False, True, False, False, True
        ])
        self.assertTrue(tree.is_densely_associated(block, fnr_thresh=0.6))
        self.assertFalse(tree.is_densely_associated(block, fnr_thresh=0.1))
    
    '''Test gini impurity functions
    Example taken from: https://towardsdatascience.com/gini-impurity-measure-dbd3878ead33
    '''
    def test_calc_gini_impurity1(self):
        ar = [True, True, False]
        self.assertAlmostEqual(tree.calc_gini_impurity(ar), 2 * 2/3 * 1/3)
    
    def test_calc_gini_impurity2(self):
        ar = [False] * 3
        self.assertEqual(tree.calc_gini_impurity(ar), 0)
    
    def test_calc_weighted_gini_impurity(self):
        lists = [[False, True, False], [True, False, True, False, True]]
        exp_result = 3/8 * 2 * 2/3 * 1/3 + 5/8 * 2 * 3/5 * 2/5
        self.assertAlmostEqual(tree.calc_weighted_gini_impurity(lists), exp_result)


