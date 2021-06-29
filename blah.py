from pathlib import Path

import pandas as pd

from config import TUNING_RESULTS_FOLDER


def main():
    TUNING_RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
    result_csvs = TUNING_RESULTS_FOLDER.glob('*.csv')
    dfs = []
    for csv in result_csvs:
        dfs.append(pd.read_csv(csv))

    full_results = pd.concat(dfs)
    full_results.to_csv(Path(TUNING_RESULTS_FOLDER, 'full_results.csv'), index=False)


if __name__ == '__main__':
    main()
