import numpy as np
from pathlib import Path
from typing import List, NamedTuple, Tuple, Iterable, Callable, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import get_scorer

from config import TEST_RESULTS_FOLDER, RAW_TEST_PREDS_FILENAME, get_columns_that_start_with, \
    CONDITION_PREFIX, INTERVENTION_PREFIX, REGRESSOR_RENAMES, SCORING, TEST_COL_RENAMES, \
    DUMMY_REGRESSOR_RENAMES
from data.data_config import NUM_SAE


class HeatMapInfo(NamedTuple):
    prefix: str
    k: int
    col_name: str


PREFIXS_TO_VISUALIZE: List[HeatMapInfo] = [HeatMapInfo(CONDITION_PREFIX, 10, 'Condition'),
                                           HeatMapInfo(INTERVENTION_PREFIX, 10, 'Intervention')]


def drop_all_cols_but(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.drop(columns=df.columns.difference(cols))


def drop_cols_in_multiple_categories(df: pd.DataFrame, df_prefix_cols_only: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    df_single_category = df[df_prefix_cols_only.sum(1) < 2].copy()
    df_prefix_cols_only = df_prefix_cols_only[df_prefix_cols_only.sum(1) < 2]
    return df_single_category, df_prefix_cols_only


def get_cat_val_from_one_hot(df_prefix_cols_only: pd.DataFrame) -> pd.Series:
    return df_prefix_cols_only.idxmax(axis=1)


def get_top_k_cat_col_from_one_hot(df_single_category: pd.DataFrame,
                                   df_prefix_cols_only: pd.DataFrame, prefix: str, col_name: str,
                                   k: int) -> pd.DataFrame:
    top_k_cat_vals = get_top_k_cat_values(df_prefix_cols_only, k)
    df_single_category[col_name] = get_cat_val_from_one_hot(df_prefix_cols_only)
    df_single_category = keep_top_k_categories(df_single_category, col_name, top_k_cat_vals)
    df_single_category[col_name] = remove_prefixes(df_single_category, prefix, col_name,
                                                   top_k_cat_vals)
    return df_single_category


def get_top_k_cat_values(df_prefix_cols_only: pd.DataFrame, k: int) -> List[str]:
    top_k_counts = df_prefix_cols_only.sum().sort_values(ascending=False)[:k]
    top_k_cat_vals = list(top_k_counts.index)
    return top_k_cat_vals


def keep_top_k_categories(df_single_category: pd.DataFrame, col_name: str,
                          top_k_cat_vals: List[str]) -> pd.DataFrame:
    return df_single_category[df_single_category[col_name].isin(top_k_cat_vals)].copy()


def load_test_results() -> pd.DataFrame:
    return pd.read_csv(Path(TEST_RESULTS_FOLDER, RAW_TEST_PREDS_FILENAME))


def mean_error_by_cat_val(df_single_category: pd.DataFrame,
                          cat_col_name: str,
                          regressor_names: Iterable[str],
                          scorer_names: Iterable[str]) -> Dict[str, pd.DataFrame]:
    to_plots = {}
    for regressor_name in regressor_names:
        score_series = []
        for scorer_name in scorer_names:
            scorer = get_scorer(scorer_name)._score_func
            col_name = '{}'.format(TEST_COL_RENAMES[scorer_name])
            score = df_single_category.groupby(by=[cat_col_name]).apply(lambda x: scorer(x[NUM_SAE], x[regressor_name])).rename(col_name)
            score_series.append(score)
        to_plots[regressor_name] = pd.concat(score_series, axis=1)
    return to_plots


def plot_heatmap(to_plot: pd.DataFrame, col_name: str, regressor_name: str) -> None:
    sns.heatmap(to_plot, cmap='RdBu_r', vmin=0.0, vmax=1.0)
    plt.xlabel('Measure')
    plt.savefig(Path(TEST_RESULTS_FOLDER, 'heatmap_{}_{}.png'.format(col_name, regressor_name)),
                bbox_inches='tight', format='png')
    plt.clf()


def remove_prefix(s: str, prefix: str):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def remove_prefixes(df_single_category: pd.DataFrame, prefix: str, col_name: str,
                    top_k_cat_vals: List[str]):
    return df_single_category[col_name].replace(dict(zip(top_k_cat_vals,
                                                         [remove_prefix(col, prefix) for col in
                                                          top_k_cat_vals])))


def rescale_series(series: pd.Series, min_v: float, max_v: float) -> pd.Series:
    return (series - min_v) / (max_v - min_v)


def rescale_all(to_plots: Dict[str, pd.DataFrame]):
    measures = [value for value in TEST_COL_RENAMES.values() if not value == 'Regressor']
    mins = {value: float("inf") for value in measures}
    maxs = {value: float("-inf") for value in measures}
    for regressor_name, to_plot in to_plots.items():
        to_plot['R^2'] = to_plot['R^2'] * -1
        curr_mins = to_plot.min()
        curr_maxs = to_plot.max()
        for measure in measures:
            if curr_mins[measure] < mins[measure]:
                mins[measure] = curr_mins[measure]
            if curr_maxs[measure] > maxs[measure]:
                maxs[measure] = curr_maxs[measure]
    for regressor_name, to_plot in to_plots.items():
        for measure in measures:
            to_plot[measure] = rescale_series(to_plot[measure], mins[measure], maxs[measure])
        to_plot.rename(columns={'R^2': '-R^2'}, inplace=True)
    return to_plots


def main():
    TEST_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.autolayout"] = True
    df = load_test_results()
    renames = {**REGRESSOR_RENAMES, **DUMMY_REGRESSOR_RENAMES}
    df = df.rename(columns=renames)

    for prefix, k, col_name in PREFIXS_TO_VISUALIZE:
        cols = get_columns_that_start_with(df, prefix)
        df_prefix_cols_only = drop_all_cols_but(df, cols)
        df_single_category, df_prefix_cols_only = drop_cols_in_multiple_categories(df,
                                                                                   df_prefix_cols_only)
        df_single_category = get_top_k_cat_col_from_one_hot(df_single_category, df_prefix_cols_only,
                                                            prefix, col_name, k)
        to_plots = mean_error_by_cat_val(df_single_category,
                                        col_name,
                                        renames.values(),
                                        SCORING)
        to_plots = rescale_all(to_plots)
        for regressor_name, to_plot in to_plots.items():
            if to_plot.index.name == 'Condition':
                to_plot.index.name = 'Intervention'
                col_name = 'Intervention'
            elif to_plot.index.name == 'Intervention':
                to_plot.index.name = 'Condition'
                col_name = 'Condition'
            plot_heatmap(to_plot, col_name, regressor_name)


if __name__ == '__main__':
    main()
