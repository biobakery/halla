import os

# associations = ['parabola', 'mixed', 'log', 'sine', 'step']
# metrics = ['dcor', 'nmi', 'spearman', 'nmi', 'dcor']
# noise_within = ['0.15', '0.15', '0.3', '0.1', '0.15']
# noise_between = ['0.15', '0.25', '0.15', '0.1', '0.25']
# iters = [20, 20, 50, 20, 20]

# 07/28
# associations = ['parabola', 'mixed', 'sine', 'step']
# metrics = ['dcor', 'nmi', 'nmi', 'dcor']
# noise_within = ['0.15', '0.15', '0.1', '0.15']
# noise_between = ['0.15', '0.25', '0.1', '0.25']
# iters = [10, 10, 10, 10]
# fdrs = ['0.05', '0.1', '0.25', '0.5']

# 07/29
associations = ['mixed', 'sine', 'step']
metrics = ['nmi', 'nmi', 'dcor']
noise_within = ['0.3', '0.25', '0.3']
noise_between = ['0.3', '0.3', '0.3']
fdrs = ['0.05', '0.1', '0.25', '0.5']


for fdr in fdrs:
    for i in range(len(associations)):
        for iter_i in range(0, 20):
            text = \
'''#!/bin/bash
#SBATCH -c 4                 # Number of cores per task
#SBATCH -N 1                 # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-4:00            # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared            # Partition to submit to
#SBATCH --mem=20G            # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o 07-30-2_%s_fdr%s_%d_''' % (associations[i], fdr, iter_i) + '''%j.out  # File to which STDOUT will be written
#SBATCH -e ''' + '07-30-2_%s_fdr%s_%d_' % (associations[i], fdr, iter_i) + '''%j.err  # File to which STDERR will be written
#SBATCH --mail-user=kathleen_sucipto@hms.harvard.edu
#SBATCH --mail-type=END,FAIL
module load Anaconda3/2019.10
module load R
source activate halla

''' + 'python examples/run_simulation.py -i 1 -n 50 -xf 200 -yf 200 --association %s --metric %s --fdr_alpha %s -nw %s -nb %s -o fin_07-30-2_%s_fdr%s_%d' % (
            associations[i], metrics[i], fdr, noise_within[i], noise_between[i], associations[i], fdr, iter_i)

            f = open('bash_%s_fdr%s_%d.sh' % (associations[i], fdr, iter_i), 'w')
            f.write(text)
            f.close()
            # print('bash_%s_fdr%s_%d.sh' % (associations[i], fdr, iter_i))
            os.system('sbatch bash_%s_fdr%s_%d.sh' % (associations[i], fdr, iter_i))