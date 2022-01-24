from pathlib import Path
from typing import List

import pandas as pd

from config import TUNING_RESULTS_FOLDER


def main():
    TUNING_RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
    result_csv_paths = TUNING_RESULTS_FOLDER.glob('*.csv')
    dfs: List[pd.DataFrame] = [pd.read_csv(csv) for csv in result_csv_paths]

    full_results = pd.concat(dfs)
    full_results.to_csv(Path(TUNING_RESULTS_FOLDER, 'full_results.csv'), index=False)


if __name__ == '__main__':
    main()
