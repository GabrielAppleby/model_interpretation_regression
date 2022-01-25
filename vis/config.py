from pathlib import Path
from typing import Dict, Tuple, Callable

from sklearn.base import RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from data_loader import load_cali_housing, load_mpg

TOP_LEVEL_FOLDER: Path = Path(__file__).parent
RESULTS_FOLDER: Path = Path(TOP_LEVEL_FOLDER, 'results')
TUNING_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'tuning')
TEST_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'test')
SAVED_MODEL_FOLDER: Path = Path(TOP_LEVEL_FOLDER, 'saved_models')

NAME_TEMPLATE = '{model}_{data}'

TEST_RESULTS_TEMPLATE: str = '{data}_test_results.csv'
RAW_TEST_PREDS_TEMPLATE: str = '{data}_raw_preds_and_truth.csv'

SCORING = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

KNN_NAME = "KNN"
XGB_NAME = "XGB"

CALI_NAME = "CALI"
MPG_NAME = "MPG"

KNN_PARAMS: Dict = {"kneighborsregressor__n_neighbors": [1] + list(range(2, 42, 4))}
XGB_PARAMS: Dict = {
    "xgbregressor__alpha": [.01, .1, .3, .75, 1.25],
    "xgbregressor__lambda": [.01, .1, .3, .75, 1.25],
    "xgbregressor__gamma": [.01, .1, .3, .75, 1.25],
    "xgbregressor__min_child_weight": [1, 2, 4],
    "xgbregressor__subsample": [.9, .95, 1],
    "xgbregressor__max_depth": [4, 8, 10, 12],
    "xgbregressor__tree_method": ["gpu_hist"]
}

REGRESSORS: Dict[str, Tuple[RegressorMixin, Dict, str]] = {
    KNN_NAME: (KNeighborsRegressor(), KNN_PARAMS, KNN_NAME),
    XGB_NAME: (XGBRegressor(), XGB_PARAMS, XGB_NAME)
}

DATASETS: Dict[str, Tuple[Callable, str]] = {CALI_NAME: (load_cali_housing, CALI_NAME),
                                             MPG_NAME: (load_mpg, MPG_NAME)}

RANDOM_SEED = 42
