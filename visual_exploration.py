from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_FOLDER, get_dummy_columns
from data.data_loader import get_shortened_dataframe

EXPLORATION_RESULTS_FOLDER: Path = Path(RESULTS_FOLDER, 'exploration')


def create_pairplot(df: pd.DataFrame, dummy_columns: List[str]) -> None:
    to_plot = df.drop(columns=dummy_columns)
    sns.pairplot(to_plot)
    plt.savefig(Path(EXPLORATION_RESULTS_FOLDER, 'pairplot.png'), bbox_inches='tight', format='png')
    plt.clf()


def create_corr_matrix(df: pd.DataFrame, dummy_columns: List[str]) -> None:
    to_plot = df.drop(columns=dummy_columns)
    corr = to_plot.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(corr, k=1)] = True
    sns.heatmap(corr, mask=mask, cmap='inferno')
    plt.savefig(Path(EXPLORATION_RESULTS_FOLDER, 'correlation_matrix.png'), bbox_inches='tight',
                format='png')
    plt.clf()


def main():
    EXPLORATION_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.autolayout"] = True
    train = get_shortened_dataframe('train')
    dummy_columns = get_dummy_columns(train)
    create_pairplot(train, dummy_columns)
    create_corr_matrix(train, dummy_columns)


if __name__ == '__main__':
    main()
