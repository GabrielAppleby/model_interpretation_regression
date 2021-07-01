from pathlib import Path
from typing import List

import pandas as pd

RESULTS_FOLDER: Path = Path(Path(__file__).parent, 'results')
TUNING_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'tuning')


def get_dummy_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if
            col.startswith('condition_') or col.startswith('intervention_') or col.startswith(
                'phase_')]
