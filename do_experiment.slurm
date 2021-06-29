#!/usr/bin/env bash
#SBATCH -n 8                # Number of cores
#SBATCH -t 0-05:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 1000  # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children


# Pass along all ENV variables as arguments to my Python script
python model_search.py --transform $transform --regressor $reg

