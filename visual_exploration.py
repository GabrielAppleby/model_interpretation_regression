from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data.dataloader import get_dataframes

EXPLORATION_RESULTS_FOLDER: Path = Path(Path(__file__).parent, 'exploration_results')
X_TRAIN = 'x_train.csv'
Y_TRAIN = 'y_train.csv'
CATEGORICAL_COLUMNS = ['has_exp_access', 'p_early_1', 'p_1', 'p_1/2', 'p_2', 'p_2/3', 'p_3', 'p_4']


def create_hists(df: pd.DataFrame, bin_sizes: List[int]):
    for bin_size in bin_sizes:
        df.hist(bins=bin_size)
        plt.savefig(Path(EXPLORATION_RESULTS_FOLDER, 'hist_{}.png'.format(bin_size)),
                    bbox_inches='tight', format='png')
        plt.clf()


def create_pairplot(df: pd.DataFrame):
    to_plot = df.drop(columns=CATEGORICAL_COLUMNS)
    sns.pairplot(to_plot)
    plt.savefig(Path(EXPLORATION_RESULTS_FOLDER, 'pairplot.png'), bbox_inches='tight', format='png')
    plt.clf()


def create_corr_matrix(df):
    to_plot = df.drop(columns=CATEGORICAL_COLUMNS)
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
    train = get_dataframes('train')
    train['sae_by_enroll_by_dur'] = train['sae_by_enroll_by_dur'] / train['enrollment'] / train[
        'actual_dur']
    create_hists(train, [5, 10, 15])
    create_pairplot(train)
    create_corr_matrix(train)


if __name__ == '__main__':
    main()
