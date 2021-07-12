from pathlib import Path
from typing import List, NamedTuple, Tuple, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import TEST_RESULTS_FOLDER, RAW_TEST_PREDS_FILENAME, get_columns_that_start_with, \
    CONDITION_PREFIX, INTERVENTION_PREFIX
from data.data_config import NUM_SAE


class HeatMapInfo(NamedTuple):
    prefix: str
    k: int
    col_name: str


REGRESSOR_RENAMES = {'xgbregressor': 'XGB',
                     'kneighborsregressor': 'KNN',
                     'linearregression': 'OLS',
                     'svr': 'SVM'}

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
    df_single_category[col_name] = remove_prefix(df_single_category, prefix, col_name,
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


def mean_error_by_cat_val(df_single_category: pd.DataFrame, col_name: str,
                          regressor_names: Iterable[str]):
    for regressor in regressor_names:
        df_single_category[regressor] = squared_error(df_single_category[NUM_SAE],
                                                      df_single_category[regressor])
    grp = df_single_category.groupby(by=[col_name]).mean()
    to_plot = grp.drop(columns=grp.columns.difference(REGRESSOR_RENAMES.values()))
    return to_plot


def plot_heatmap(to_plot: pd.DataFrame, col_name: str) -> None:
    sns.heatmap(to_plot, cmap=sns.cm.rocket_r, cbar_kws={'label': 'Mean Squared Error'})
    plt.xlabel('Regressor')
    plt.savefig(Path(TEST_RESULTS_FOLDER, 'heatmap_{}.png'.format(col_name)),
                bbox_inches='tight', format='png')
    plt.clf()


def remove_prefix(df_single_category: pd.DataFrame, prefix: str, col_name: str,
                  top_k_cat_vals: List[str]):
    return df_single_category[col_name].replace(dict(zip(top_k_cat_vals,
                                                         [col[len(prefix):] for col in
                                                          top_k_cat_vals])))


def squared_error(true: pd.Series, pred: pd.Series) -> pd.Series:
    return (true - pred) ** 2


def main():
    TEST_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.autolayout"] = True
    df = load_test_results()
    df = df.rename(columns=REGRESSOR_RENAMES)

    for prefix, k, col_name in PREFIXS_TO_VISUALIZE:
        cols = get_columns_that_start_with(df, prefix)
        df_prefix_cols_only = drop_all_cols_but(df, cols)
        df_single_category, df_prefix_cols_only = drop_cols_in_multiple_categories(df,
                                                                                   df_prefix_cols_only)
        df_single_category = get_top_k_cat_col_from_one_hot(df_single_category, df_prefix_cols_only,
                                                            prefix, col_name, k)
        to_plot = mean_error_by_cat_val(df_single_category, col_name, REGRESSOR_RENAMES.values())
        plot_heatmap(to_plot, col_name)


if __name__ == '__main__':
    main()
