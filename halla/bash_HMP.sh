#!/bin/bash
#SBATCH -c 4                 # Number of cores per task
#SBATCH -N 1                 # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-8:00            # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared            # Partition to submit to
#SBATCH --mem=20G            # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o HMP_%j.out  # File to which STDOUT will be written
#SBATCH -e HMP_%j.err  # File to which STDERR will be written
#SBATCH --mail-user=kathleen_sucipto@hms.harvard.edu
#SBATCH --mail-type=END,FAIL
module load Anaconda3/2019.10
module load R
source activate halla

python examples/run_HMP.py