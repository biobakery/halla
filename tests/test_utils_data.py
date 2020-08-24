import unittest
import numpy as np
import pandas as pd
from scipy.stats import zscore
from halla.utils import data

from utils import compare_numpy_array

class TestDataUtils(unittest.TestCase):

    '''Tests on eval_type functions
    '''
    def test_eval_type_input_df(self):
        '''Tests if it throws an error when non-pd.DataFrame arg is given
        '''
        self.assertRaises(ValueError, data.eval_type, np.zeros((5, 5)))
    
    def test_eval_type_result_1(self):
        df = pd.DataFrame(data={ 'sample_1': ['24', 'a'],
                                 'sample_2': ['13.6', 'b'],
                                 'sample_3': [np.nan, 'a'] })
        _, types = data.eval_type(df)
        self.assertTrue(compare_numpy_array(types, np.array([float, object])))
    
    def test_eval_type_result_2(self):
        df = pd.DataFrame(data={ 'sample_1': ['24', 'a'],
                                 'sample_2': ['13.6', 'b'],
                                 'sample_3': [np.nan, 'a'] })
        updated_df, _ = data.eval_type(df)
        self.assertTrue(compare_numpy_array(updated_df.iloc[0].to_numpy(), np.array([24, 13.6, np.nan])))
    
    def test_eval_type_result_3(self):
        df = pd.DataFrame(data={ 'sample_1': ['24', 'a'],
                                 'sample_2': ['13.6', 'b'],
                                 'sample_3': [np.nan, 'a'] })
        updated_df, _ = data.eval_type(df)
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

    def test_dicretize_vector_continuous_quantile(self):
        ar = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, np.nan, 4, 4, 4, 4, 5, 5, 5, 6, 6, np.nan])
        res = data.discretize_vector(ar, func='quantile', num_bins=3)
        expected_res = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])
        self.assertTrue(compare_numpy_array(res, expected_res))

    def test_dicretize_vector_continuous_jenks(self):
        ar = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, np.nan, 4, 4, 4, 4, 5, 5, 5, 6, 6, np.nan])
        res = data.discretize_vector(ar, func='jenks', num_bins=3)
        expected_res = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])
        self.assertTrue(compare_numpy_array(res, expected_res))
    
    '''Tests on keep_feature function
    '''
    def test_keep_feature_1(self):
        ar = [1, 1, 0, 0, 0, 0, 0, 1, 0, 1]
        df_row = pd.DataFrame(ar).T.iloc[0]
        self.assertFalse(data.keep_feature(df_row, 0.5))
    
    def test_keep_feature_2(self):
        ar = [1, 1, 0, 0, 0, 0, 0, 1, 0, 1]
        df_row = pd.DataFrame(ar).T.iloc[0]
        self.assertFalse(data.keep_feature(df_row, 0.6))
    
    def test_keep_feature_3(self):
        ar = [1, 1, 0, 0, 0, 0, 0, 1, 0, 1]
        df_row = pd.DataFrame(ar).T.iloc[0]
        self.assertTrue(data.keep_feature(df_row, 0.61))
    
    def test_keep_feature_4(self):
        ar = [1, 1, 1, 1, 1, 1, 1, 1]
        df_row = pd.DataFrame(ar).T.iloc[0]
        self.assertTrue(data.keep_feature(df_row, None))
    
    def test_keep_feature_5(self):
        ar = [1, 1, 1, 1, 1, 1, 1, 1]
        df_row = pd.DataFrame(ar).T.iloc[0]
        self.assertFalse(data.keep_feature(df_row, 1))
    
    '''Tests on transform function
    '''
    def test_transform_invalid_func(self):
        '''Tests if it throws an error when an invalid function is given
        '''
        self.assertRaises(ValueError, data.transform, None, None, 'strange_func')

    def test_transform_log_1(self):
        df = pd.DataFrame(data={ 'sample_1': [24, 'a'],
                                 'sample_2': [13.6, 'b'],
                                 'sample_3': [np.nan, 'a'] })
        types = np.array([float, object])
        updated_df = data.transform(df, types, ['log'])
        self.assertTrue(compare_numpy_array(updated_df.iloc[0].to_numpy(), np.log([24, 13.6, np.nan])))
    
    def test_transform_log_2(self):
        df = pd.DataFrame(data={ 'sample_1': [24, 'a'],
                                 'sample_2': [13.6, 'b'],
                                 'sample_3': [np.nan, 'a'] })
        types = np.array([float, object])
        updated_df = data.transform(df, types, ['log'])
        self.assertTrue(compare_numpy_array(updated_df.iloc[1].to_numpy(), np.array(['a', 'b', 'a'])))
    
    def test_transform_log_3(self):
        df = pd.DataFrame(data={ 'sample_1': [24, 'a'],
                                 'sample_2': [-13.6, 'b'],
                                 'sample_3': [np.nan, 'a'] })
        types = np.array([float, object])
        self.assertRaises(ValueError, data.transform, df, types, ['log'])
    
    def test_transform_sqrt(self):
        df = pd.DataFrame(data={ 'sample_1': [24, 'a'],
                                 'sample_2': [-13.6, 'b'],
                                 'sample_3': [np.nan, 'a'] })
        types = np.array([float, object])
        row = np.array([24, -13.6, np.nan])
        updated_df = data.transform(df, types, ['sqrt'])
        self.assertTrue(compare_numpy_array(updated_df.iloc[0].to_numpy(), np.sqrt(np.abs(row)) * np.sign(row)))
    
    def test_transform_multiple_functions(self):
        df = pd.DataFrame(data={ 'sample_1': [24, 'a'],
                                 'sample_2': [-13.6, 'b'],
                                 'sample_3': [np.nan, 'a'],
                                 'sample_4': [10, 'a'] })
        types = np.array([float, object])
        updated_df = data.transform(df, types, ['zscore', 'sqrt'])
        expected_res = zscore(np.array([24, -13.6, np.nan, 10]), nan_policy='omit')
        expected_res = np.sqrt(np.abs(expected_res)) * np.sign(expected_res)
        self.assertTrue(compare_numpy_array(updated_df.iloc[0].to_numpy(), expected_res))
    
    