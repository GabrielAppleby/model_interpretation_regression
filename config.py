from pathlib import Path
from typing import List

import pandas as pd

TOP_LEVEL_FOLDER: Path = Path(__file__).parent
RESULTS_FOLDER: Path = Path(TOP_LEVEL_FOLDER, 'results')
TUNING_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'tuning')
TEST_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'test')
SAVED_MODEL_FOLDER: Path = Path(TOP_LEVEL_FOLDER, 'saved_models')

TEST_RESULTS_FILENAME: str = 'test_results.csv'
RAW_TEST_PREDS_FILENAME: str = 'raw_preds_and_truth.csv'

CONDITION_PREFIX = 'condition_mesh_term_'
INTERVENTION_PREFIX = 'intervention_mesh_term_'
EXPANDED_ACCESS_PREFIX = 'has_exp'
PHASE_PREFIX = 'phase_'
SCORING = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
DUMMY_PREFIXES = [CONDITION_PREFIX, INTERVENTION_PREFIX, PHASE_PREFIX, EXPANDED_ACCESS_PREFIX]

REGRESSOR_RENAMES = {'xgbregressor': 'XGB',
                     'kneighborsregressor': 'KNN',
                     'linearregression': 'OLS',
                     'svr': 'SVM'}

TEST_COL_RENAMES = {'neg_mean_absolute_error': 'MAE',
                    'neg_mean_squared_error': 'MSE',
                    'r2': 'R^2',
                    'name': 'Regressor'}

DUMMY_REGRESSOR_RENAMES = {'dummyregressor_mean': 'Mean', 'dummyregressor_median': 'Median'}


def get_dummy_columns(df: pd.DataFrame) -> List[str]:
    columns = []
    for prefix in DUMMY_PREFIXES:
        columns.extend(get_columns_that_start_with(df, prefix))
    return columns


def get_columns_that_start_with(df: pd.DataFrame, starts_with: str):
    return [col for col in df.columns if col.startswith(starts_with)]
