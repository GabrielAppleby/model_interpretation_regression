from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_FOLDER, DATASETS
from data_loader import DataSplit

EXPLORATION_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'exploration')


def create_pairplot(df: pd.DataFrame, data_name: str) -> None:
    sns.pairplot(df)
    plt.savefig(Path(EXPLORATION_RESULTS_FOLDER, '{}_pairplot.png'.format(data_name)),
                bbox_inches='tight', format='png')
    plt.clf()


def create_corr_matrix(df: pd.DataFrame, data_name: str) -> None:
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(corr, k=1)] = True
    sns.heatmap(corr, mask=mask, cmap='inferno', cbar_kws={'label': 'Correlation'})
    plt.xlabel('Column')
    plt.ylabel('Column')
    plt.savefig(Path(EXPLORATION_RESULTS_FOLDER, '{}_correlation_matrix.png'.format(data_name)),
                bbox_inches='tight',
                format='png')
    plt.clf()


def main():
    EXPLORATION_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.autolayout"] = True
    for dataset_fnc, data_name in DATASETS.values():
        train: DataSplit = dataset_fnc(42).train
        full_df = pd.concat([train.features.reset_index(drop=True),
                             train.targets.reset_index(drop=True)], axis=1)
        create_pairplot(full_df, data_name)
        create_corr_matrix(full_df, data_name)


if __name__ == '__main__':
    main()
