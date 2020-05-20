#TODO: move to the appropriate directory

from tools import HAllA

test_halla = HAllA('dummy_param')
test_halla.load('data/X_line1_32_50.txt', 'data/Y_line1_32_50.txt')
test_halla.run()