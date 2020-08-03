import unittest
import sys
from os.path import dirname, abspath
import numpy as np
import pandas as pd

sys.path.append(dirname(dirname(abspath(__file__))))

from tools.utils import data
from utils import compare_numpy_array

class TestDataUtils(unittest.TestCase):

    '''Tests on eval_type functions
    '''
    def test_eval_type_input_df(self):
        '''Tests if it throws an error when non-pd.DataFrame arg is given
        '''
        self.assertRaises(ValueError, data.eval_type, np.zeros((5, 5)))
    
    def test_eval_type_result(self):
        '''Tests the results
        '''
        df = pd.DataFrame(data={ 'sample_1': ['24', 'a'],
                                 'sample_2': ['13.6', 'b'],
                                 'sample_3': [None, 'a'] })
        updated_df, types = data.eval_type(df)
        self.assertTrue(compare_numpy_array(types, np.array([float, object])))
        self.assertTrue(compare_numpy_array(updated_df.iloc[0].to_numpy(), np.array([24, 13.6, np.nan])))
        self.assertTrue(compare_numpy_array(updated_df.iloc[1].to_numpy(), np.array(['a', 'b', 'a'])))

    '''Tests on discretization
    '''
    def test_discretize_vector_categorical(self):
        ar = np.array(['a', 'c', 'b', 'b', 'c', 'b', 'a', 'd', 'a', 'd', 'e'])
        res = data.discretize_vector(ar, ar_type=object)
        expected_res = np.array([0, 1, 2, 2, 1, 2, 0, 3, 0, 3, 4])
        self.assertTrue(compare_numpy_array(res, expected_res))
    
    def test_discretize_vector_categorical_missing_data(self):
        ar = np.array(['a', 'c', 'b', 'b', 'c', np.nan, 'a', 'd', 'a', 'd', 'e', np.nan])
        res = data.discretize_vector(ar, ar_type=object)
        expected_res = np.array([0, 1, 2, 2, 1, 3, 0, 4, 0, 4, 5, 3])
        self.assertTrue(compare_numpy_array(res, expected_res))

    def test_dicretize_vector_continuous_equal_freq(self):
        ar = np.array([1,1,1,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6])
        res = data.discretize_vector(ar, func='equal-freq', num_bins=5)
        expected_res = pd.cut(ar, bins=5, labels=False)
        self.assertTrue(compare_numpy_array(res, expected_res))

    def test_dicretize_vector_continuous_missing_data(self):
        ar = np.array([1,1,2,2,np.nan,3,3,np.nan])
        res = data.discretize_vector(ar, func='equal-freq', num_bins=3)
        expected_res = np.array([0,0,1,1,3,2,2,3])
        self.assertTrue(compare_numpy_array(res, expected_res))