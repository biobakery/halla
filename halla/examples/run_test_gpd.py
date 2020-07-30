from os.path import dirname, abspath, join
import sys
import argparse
import pandas as pd
import time

sys.path.append(dirname(dirname(abspath(__file__))))

from tools import HAllA, AllA
from tools.utils.stats import compute_result_power, compute_result_fdr

X_file = join(dirname(abspath(__file__)), '../simulation', '07-13_fdr01_trim_nw05_0', 'X_line_500_50.txt')
Y_file = join(dirname(abspath(__file__)), '../simulation', '07-13_fdr01_trim_nw05_0', 'Y_line_500_50.txt')
A_file = join(dirname(abspath(__file__)), '../simulation', '07-13_fdr01_trim_nw05_0', 'A_line_500_500.txt')
A = pd.read_table(A_file, index_col=0).to_numpy()

# parse arguments
parser = argparse.ArgumentParser(description='Run simulations to test GPD')
parser.add_argument('-f', '--func', help='Permutation function', choices=['gpd', 'ecdf'], default='ecdf')
parser.add_argument('-o', '--output', help='Output directory', required=True)
parser.add_argument('--speedup', help='Speedup permutation', default=False, action='store_true')
params = parser.parse_args()

pdist_metric = 'spearman'

start_time = time.time()

test_halla = HAllA(pdist_metric=pdist_metric, out_dir='simulation_out/%s' % params.output,
                    permute_func=params.func, permute_iters=1000, permute_speedup=params.speedup)

test_halla.load(X_file, Y_file)
test_halla.run()

total_dur = time.time() - start_time()

power = compute_result_power(test_halla.significant_blocks, A)
fdr = compute_result_fdr(test_halla.significant_blocks, A)

pd.DataFrame(data={
    'time': [total_dur],
    'power': [power],
    'fdr': [fdr]
}).to_csv(join('simulation_out/%s' % params.output, 'result.csv'))

# test_halla.generate_hallagram()
# test_halla.generate_diagnostic_plot()