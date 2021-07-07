#!/usr/bin/env bash
#SBATCH -p batch                  # Partition to submit to
#SBATCH -t 1-00:00                # Runtime in D-HH:MM
#SBATCH --nodes 1                 # Number of nodes
#SBATCH --ntasks 1                # Number of tasks
#SBATCH --cpus-per-task 8         # Number of cpus
#SBATCH --mem-per-cpu 2000        # Memory (in MB) per cpu
#SBATCH -o log_%j.out             # Write stdout to file named log_JOBID.out in current dir
#SBATCH -e log_%j.err             # Write stderr to file named log_JOBID.err in current dir
#SBATCH --export=ALL              # Pass any exported env vars to this script and its children
#SBATCH --mail-type=fail          # send email if job fails
#SBATCH --mail-user=Gabriel.Appleby@tufts.edu

source activate model_interp

# Pass along all ENV variables as arguments to my Python script
python model_search.py $reg
