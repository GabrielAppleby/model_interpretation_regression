#!/bin/bash
#
# Usage
# -----
# $ bash launch_cluster_model_search.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'run_here'

if [[ -z $1 ]]; then
  ACTION_NAME='list'
else
  ACTION_NAME=$1
fi

for reg in "KNN" "SVR" "XGB" "LIN" "MEAN" "MEDIAN"; do

  ## Use this line to see where you are in the loop
  echo "reg=$reg"

  if [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    python model_search.py $reg
  fi

done
