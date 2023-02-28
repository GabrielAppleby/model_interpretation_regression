# README

## Requirements
- Create whatever virtual env you prefer
- Install the requirements found in requirements.txt

## Which folder do I care about?
- vahc contains the setup used for the VAHC paper
  - Most importantly, it is set up to run on data from [AACT](https://aact.ctti-clinicaltrials.org/)
  - More models are explored
- vis contains the setup used for the VIS paper
  - Well known benchmarking data is used for this
  - Only two models but a larger search space of hyperparameters is explored for the two (KNN, and XGB)

## Getting Started

I generally turn everything into individual scripts so it can get a bit hectic.

### VAHC

- Download the data from our Study 2021 google drive folder
  - AACT Regression Model -> aact
  - More information about the data is found in this folder as well as the paper.
- Place it the aact folder with vahc/data
  - If confused check the vahc/data/data_config.py for the path used
- Tuning
  - You can run launch_local_model_search.sh with the run_here argument to run through the tuning search for each model
  - Or you can run model_search.py directly with the model arg of your choice
  - This should populate results/tuning
  - Editing modeling/regressors changes the search parameters
  - After tuning each model you need to run aggregate_results.py to put them all together
    - (My computer is slow so I generally run things on the cluster using the slurm file)
- Testing
  - Run test_best_models.py to test the best models found in the search
  - This will populate results/test
- Visualizing
  - table_results.py turns the results into a latex table
  - visual_results.py creates the figures for the results found in the paper
    - *I got a nan running the plot_ellipse function*
  - visual_exploration.py lets you explore the raw data one split at a time
    - So you can explore the training data in the tuning phase when you shouldn't have access to Test

### VIS

The vis setup was created by copy and pasting the VAHC setup and simplifying, so don't plan to go through it again. 

Generally:
- As stated above larger search space for KNN and XGB
- Added a search_to_hiplot.py for looking at the data in [HiPlot](https://facebookresearch.github.io/hiplot/_static/hiplot_upload.html)
