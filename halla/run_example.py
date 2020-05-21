#TODO: move to the appropriate directory

from tools import HAllA
from os.path import dirname, abspath, join

X_file = join(dirname(abspath(__file__)), 'data', 'X_line1_32_50.txt')
Y_file = join(dirname(abspath(__file__)), 'data', 'Y_line1_32_50.txt')

test_halla = HAllA()
test_halla.load(X_file, Y_file)
test_halla.run()