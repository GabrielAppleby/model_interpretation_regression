from pathlib import Path
from typing import List, NamedTuple, Tuple, Iterable, Dict

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.pyplot import figure
from sklearn.metrics import get_scorer
from sklearn.metrics import r2_score

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
            score = df_single_category.groupby(by=[cat_col_name]).apply(
                lambda x: scorer(x[NUM_SAE], x[regressor_name])).rename(col_name)
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


def plot_ellipse(predicted: pd.Series, actual: pd.Series, model_name: str, n_std=2.0,
                 facecolor='none') -> None:
    # Taken from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    x = predicted
    y = actual
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, alpha=0.2)

    print("for model_name {}, radius x is {}, radius y is {}".format(model_name, ell_radius_x,
                                                                     ell_radius_y))
    print(" and the sklearn r2_score is ", r2_score(y, x))
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    # ellipse.set_transform(transf)

    figure(figsize=(6, 6), dpi=80)
    ax = plt.gca()
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    ax.scatter(x, y, c='red', s=3)
    ax.set_title("Correlation between predicted and actual, {}".format(model_name))

    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.xlim((-1.0, 6.0))
    plt.ylim((-1.0, 6.0))

    plt.savefig(Path(TEST_RESULTS_FOLDER, 'scatter_{}.png'.format(model_name)),
                bbox_inches='tight', format='png')
    plt.clf()


def plot_residuals(predicted: pd.Series, actual: pd.Series, model_name: str) -> None:
    # Taken from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    x = predicted
    y = actual
    errors = np.abs(x - y)
    mean_error = np.mean(errors)
    sorted_errors = np.sort(errors)[::-1]
    # rectangle = Rectangle((0, 0), width=6.0, height=mean_error, alpha=0.2)
    rectangle = Rectangle((0, 0), width=1000.0, height=mean_error, alpha=0.2, color='green')

    figure(figsize=(6, 6), dpi=80)
    ax = plt.gca()
    ax.add_patch(rectangle)
    ax.bar(range(len(sorted_errors)), sorted_errors)
    ax.annotate("Mean Absolute Error: {:0.3f}".format(mean_error), (460.0, 1.5))
    # ax.annotate("Mean Absolute Error: {:0.3f}".format(mean_error), (600.0, 1.5))
    # ellipse.set_transform(transf + ax.transData)
    # ax.add_patch(ellipse)
    # ax.scatter(x, y, c='red', s=3)
    ax.set_title("Absolute error per site, {}".format(model_name))

    plt.xlabel('sites sorted by error')
    plt.ylabel('Absolute error')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    # plt.xlim((-1.0, 6.0))
    plt.ylim((0.0, 12.0))
    # plt.ylim((0.0, 13.0))

    plt.savefig(Path(TEST_RESULTS_FOLDER, 'residuals_{}.png'.format(model_name)),
                bbox_inches='tight', format='png')
    plt.clf()


def main():
    TEST_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.autolayout"] = True
    df = load_test_results()
    renames = {**REGRESSOR_RENAMES, **DUMMY_REGRESSOR_RENAMES}
    df = df.rename(columns=renames)
    sns.set(font_scale=1)

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

    sns.reset_orig()
    plt.rcParams.update({'font.size': 12})
    for model_name in ['XGB', 'Mean', 'OLS', 'Median', 'SVM', 'KNN']:
        plot_ellipse(df[model_name], df.number_of_sae_subjects, model_name)
        plot_residuals(df[model_name], df.number_of_sae_subjects, model_name)


if __name__ == '__main__':
    main()
