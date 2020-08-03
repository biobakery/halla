import os

# associations = ['parabola', 'mixed', 'log', 'sine', 'step']
# metrics = ['dcor', 'nmi', 'spearman', 'nmi', 'dcor']
# noise_within = ['0.15', '0.15', '0.3', '0.1', '0.15']
# noise_between = ['0.15', '0.25', '0.15', '0.1', '0.25']
# iters = [20, 20, 50, 20, 20]

associations = ['parabola', 'mixed', 'sine', 'step']
metrics = ['dcor', 'nmi', 'nmi', 'dcor']
noise_within = ['0.15', '0.35', '0.3', '0.4']
noise_between = ['0.15', '0.35', '0.35', '0.4']
iters = [20, 20, 20, 20]

for i in range(len(associations)):
    for iter_i in range(20):
        text = \
'''#!/bin/bash
#SBATCH -n 4                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-4:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=20G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o 07-30_%s_fnr_%d_''' % (associations[i], iter_i) + '''%j.out  # File to which STDOUT will be written
#SBATCH -e ''' + '07-30_%s_fnr_%d_' % (associations[i], iter_i) + '''%j.err  # File to which STDERR will be written
#SBATCH --mail-user=kathleen_sucipto@hms.harvard.edu
#SBATCH --mail-type=END,FAIL
module load Anaconda3/2019.10
module load R
source activate halla

''' + 'python examples/run_simulation_fnr.py -i 1 -n 50 -xf 200 -yf 200 --association %s --metric %s -nw %s -nb %s -o fin_07-30_fnr_%s_%d' % (
        associations[i], metrics[i], noise_within[i], noise_between[i], associations[i], iter_i)

        f = open('bash_fnr_%s.sh' % (associations[i]), 'w')
        f.write(text)
        f.close()
        os.system('sbatch bash_fnr_%s.sh' % (associations[i]))