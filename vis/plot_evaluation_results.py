from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vis.config import RESULTS_FOLDER

# Remove first for underlying plot.
TAB_9_PALETTE = ['#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2',
                  '#7f7f7f', '#bcbd22', '#17becf']

GUIDELINE_EVALUATION_PATH = Path(RESULTS_FOLDER, 'guideline_evaluation')

sns.set(rc={'figure.figsize':(3.48,  3.48), 'figure.dpi': 300, 'savefig.dpi': 300})


def main():
    df = pd.read_csv(Path(GUIDELINE_EVALUATION_PATH, 'evaluation_results.csv'))
    df = pd.melt(df, 'PID', var_name='Difference', value_name='Score')
    sns.barplot(x=df['Difference'], y=df['Score'], color="#1f77b4", ci=None)
    # sns.boxplot(x=df['Difference'], y=df['Score'], color="#1f77b4")
    # sns.swarmplot(x=df['Difference'], y=df['Score'], hue=df['PID'], palette=TAB_9_PALETTE)
    plt.savefig(Path(GUIDELINE_EVALUATION_PATH, 'evaluation_results.pdf'), bbox_inches='tight',
                format='pdf')


if __name__ == '__main__':
    main()
