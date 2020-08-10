import unittest
import sys
from os.path import dirname, abspath
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, ClusterNode
from halla.utils import tree

from utils import compare_numpy_array

def generate_test_x_and_y_trees():
    # create X features: [[0, 1], [2, 3]]
    X0 = ClusterNode(4, ClusterNode(0, None, None), ClusterNode(1, None, None))
    X1 = ClusterNode(5, ClusterNode(2, None, None), ClusterNode(3, None, None))
    x_tree = ClusterNode(6, X0, X1)

    # create Y features: [[0], [1, 2]]
    Y0 = ClusterNode(0, None, None)
    Y1 = ClusterNode(3, ClusterNode(1, None, None), ClusterNode(2, None, None))
    y_tree = ClusterNode(5, Y0, Y1)
    return(x_tree, y_tree)

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

    def test_calc_gini_impurity3(self):
        # has to be a boolean array, not just binary
        self.assertRaises(ValueError, tree.calc_gini_impurity, [0, 1])
    
    def test_calc_weighted_gini_impurity(self):
        lists = [[False, True, False], [True, False, True, False, True]]
        exp_result = 3/8 * 2 * 2/3 * 1/3 + 5/8 * 2 * 3/5 * 2/5
        self.assertAlmostEqual(tree.calc_weighted_gini_impurity(lists), exp_result)

    '''Test bifurcation functions
    Example taken from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    '''
    def test_bifurcate1(self):
        X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
        test_tree = to_tree(linkage(X, 'ward'))
        branches = [branch.pre_order() for branch in tree.bifurcate(test_tree)]
        self.assertListEqual(branches, [[1,5,6], [3,2,7,0,4]])
    
    def test_bifurcate2(self):
        X = [[i] for i in [2, 8]]
        test_tree = to_tree(linkage(X, 'ward'))
        left_branch = tree.bifurcate(test_tree)[0]
        self.assertListEqual(tree.bifurcate(left_branch), [])

    def test_bifurcate_one_case1(self):
        '''If both are leaves, return (None, None)
        '''
        X = [[i] for i in [2, 8]]
        test_tree = to_tree(linkage(X, 'ward'))
        reject_table = np.zeros((2,2)).astype(bool)
        branches = tree.bifurcate(test_tree)
        x_tree, y_tree = branches[0], branches[1]
        self.assertEqual(tree.bifurcate_one(x_tree, y_tree, reject_table), (None, None))
    
    def test_bifurcate_one_case2_1(self):
        '''If one of them is a leaf, return the bifurcated non-leaf tree (case 1)
        '''
        X = [[i] for i in [2, 8, 1]]
        test_tree = to_tree(linkage(X, 'ward'))
        reject_table = np.zeros((3, 3)).astype(bool)
        branches = tree.bifurcate(test_tree) # --> [[1], [0,2]]
        x_tree, y_tree = branches[0], branches[1] # x_tree is a leaf
        res = tree.bifurcate_one(x_tree, y_tree, reject_table)
        # x_tree should not be bifurcated
        self.assertListEqual([x_tree.pre_order()], [branch.pre_order() for branch in res[0]])
        # y_tree should be bifurcated
        self.assertListEqual([branch.pre_order() for branch in res[1]], [[0], [2]])

    def test_bifurcate_one_case2_2(self):
        '''If one of them is a leaf, return the bifurcated non-leaf tree (case 2)
        '''
        X = [[i] for i in [1, 2, 8]]
        test_tree = to_tree(linkage(X, 'ward'))
        reject_table = np.zeros((3, 3)).astype(bool)
        branches = tree.bifurcate(test_tree) # --> [2], [0, 1]]
        x_tree, y_tree = branches[1], branches[0] # y_tree is a leaf ([2])
        res = tree.bifurcate_one(x_tree, y_tree, reject_table)
        # x_tree should be bifurcated
        self.assertListEqual([branch.pre_order() for branch in res[0]], [[0], [1]])
        # y_tree should not be bifurcated
        self.assertListEqual([y_tree.pre_order()], [branch.pre_order() for branch in res[1]])
    
    def test_bifurcate_one_case3(self):
        '''If both are non-leaves, bifurcate the one that produces lower gini impurity
        '''
        x_tree, y_tree = generate_test_x_and_y_trees()

        # create fdr reject table for X x Y = [4 x 3] table
        reject_table = np.array([
            [False, True, False],
            [False, True, True],
            [True, True, True],
            [False, False, True],
        ])
        x_branches, y_branches = tree.bifurcate_one(x_tree, y_tree, reject_table)
        # should bifurcate Y, not X
        self.assertListEqual([branch.pre_order() for branch in x_branches], [[0, 1, 2, 3]])
        self.assertListEqual([branch.pre_order() for branch in y_branches], [[0], [1, 2]])

    '''Test densely-associated block finding function
    '''
    def test_compare_and_find_dense_block1(self):
        x_tree, y_tree = generate_test_x_and_y_trees()

        reject_table = np.array([
            [False, True, False],
            [False, True, True],
            [True, True, True],
            [False, False, True],
        ])
        res = tree.compare_and_find_dense_block(x_tree, y_tree, reject_table, fnr_thresh=1)
        # with fnr thresh = 1, should return the whole block
        self.assertListEqual(res, [[[0, 1, 2, 3], [0, 1, 2]]])
    
    def test_compare_and_find_dense_block2(self):
        x_tree, y_tree = generate_test_x_and_y_trees()

        reject_table = np.array([
            [False, True, False],
            [False, True, True],
            [True, True, True],
            [False, False, True],
        ])
        res = tree.compare_and_find_dense_block(x_tree, y_tree, reject_table, fnr_thresh=0)
        self.assertListEqual(res, [[[2], [0]],
                                    [[0, 1], [1]],
                                    [[2], [1]],
                                    [[1], [2]],
                                    [[2, 3], [2]]])
    
    def test_compare_and_find_dense_block3(self):
        x_tree, y_tree = generate_test_x_and_y_trees()

        reject_table = np.array([
            [False, True, False],
            [False, True, True],
            [True, True, True],
            [False, False, True],
        ])
        res = tree.compare_and_find_dense_block(x_tree, y_tree, reject_table, fnr_thresh=0.25)
        self.assertListEqual(res, [[[2], [0]],
                                    [[0, 1, 2, 3], [1, 2]]])
    
    def test_compare_and_find_dense_block4(self):
        x_tree, y_tree = generate_test_x_and_y_trees()

        reject_table = np.asarray([False] * 12).reshape((4, 3))
        res = tree.compare_and_find_dense_block(x_tree, y_tree, reject_table, fnr_thresh=0)
        self.assertListEqual(res, [])



