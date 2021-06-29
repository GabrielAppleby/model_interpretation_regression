#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
  ACTION_NAME='list'
else
  ACTION_NAME=$1
fi

for transform in "SSCALE" "KBINS" "PTRAN"; do
  export transform=$transform

  for reg in "KNN" "SVR" "XGB" "LIN"; do
    export reg=$reg

    ## Use this line to see where you are in the loop
    echo "transform=$transform  reg=$reg"

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
      ## Use this line to submit the experiment to the batch scheduler
      sbatch <do_experiment.slurm

    elif [[ $ACTION_NAME == 'run_here' ]]; then
      ## Use this line to just run interactively
      bash do_experiment.slurm
    fi

  done
done

export transform="NONE"
for reg in "MEAN" "MEDIAN"; do
  export reg=$reg

  ## Use this line to see where you are in the loop
  echo "transform=$transform  reg=$reg"

  ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

  if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <do_experiment.slurm

  elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash do_experiment.slurm
  fi

done
