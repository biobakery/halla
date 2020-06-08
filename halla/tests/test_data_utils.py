import unittest
import sys
from os.path import dirname, abspath
import numpy as np
import pandas as pd

sys.path.append(dirname(dirname(abspath(__file__))))

import tools.utils.data as data

def compare_numpy_array(ar1, ar2):
    '''Compare and check if the values of two numpy arrays are equal
    '''
    all_float = False
    try:
        ar1 = ar1.astype(float)
        ar2 = ar2.astype(float)
        all_float = True
        np.testing.assert_allclose(ar1, ar2, verbose=True)
        return(True)
    except:
        if all_float: return(False)
        try:
            np.testing.assert_array_equal(ar1, ar2)
            return(True)
        except:
            return(False)

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

    def test_dicretize_vector_equal_freq(self):
        ar = np.array([1,1,1,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6])
        res = data.discretize_vector(ar, func='equal-freq', num_bins=5)
        expected_res = pd.cut(ar, bins=5, labels=False)
        self.assertTrue(compare_numpy_array(res, expected_res))