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
    
    