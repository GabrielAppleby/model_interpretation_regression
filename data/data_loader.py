from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from data.data_config import HAS_EXP_ACCESS, NUM_SAE, DATA_FOLDER

SHORTENED_COLUMNS = {'number_of_arms': 'num_arms',
                     'enrollment': 'enrollment',
                     HAS_EXP_ACCESS: 'has_exp_access',
                     'number_of_facilities': 'num_facil',
                     'actual_duration': 'actual_dur',
                     'months_to_report_results': 'months_to_report',
                     'minimum_age_num': 'min_age',
                     'number_of_primary_outcomes_to_measure': 'num_primary_outcomes',
                     'number_of_secondary_outcomes_to_measure': 'num_secondary_outcomes',
                     NUM_SAE: 'sae_by_enroll',
                     'count': 'count'}


def get_shortened_dataframe(split: str) -> pd.DataFrame:
    x, y = get_dataframes(split)

    df = pd.concat((x, y), axis=1)
    df = df.rename(SHORTENED_COLUMNS, axis=1)
    return df


def get_arrays(split: str) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.read_csv(Path(DATA_FOLDER, 'x_{}.csv'.format(split))).values
    y = pd.read_csv(Path(DATA_FOLDER, 'y_{}.csv'.format(split))).values.ravel()
    return x, y


def get_dataframes(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(Path(DATA_FOLDER, 'x_{}.csv'.format(split)))
    y = pd.read_csv(Path(DATA_FOLDER, 'y_{}.csv'.format(split)))
    return x, y
