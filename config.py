from pathlib import Path
from typing import List

import pandas as pd

TOP_LEVEL_FOLDER: Path = Path(__file__).parent
RESULTS_FOLDER: Path = Path(TOP_LEVEL_FOLDER, 'results')
TUNING_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'tuning')
TEST_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'test')
SAVED_MODEL_FOLDER: Path = Path(TOP_LEVEL_FOLDER, 'saved_models')
CONDITION_PREFIX = 'condition_mesh_term_'
INTERVENTION_PREFIX = 'intervention_mesh_term_'
PHASE_PREFIX = 'phase_'
SCORING = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
DUMMY_PREFIXES = [CONDITION_PREFIX, INTERVENTION_PREFIX, PHASE_PREFIX]


def get_dummy_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if
            col.startswith(CONDITION_PREFIX) or col.startswith(
                INTERVENTION_PREFIX) or col.startswith(
                PHASE_PREFIX) or col == 'has_expanded_access']
