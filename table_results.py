from pathlib import Path

import pandas as pd

from config import TEST_RESULTS_FOLDER, TEST_RESULTS_FILENAME, REGRESSOR_RENAMES, \
    DUMMY_REGRESSOR_RENAMES, TUNING_RESULTS_FOLDER, TEST_COL_RENAMES

TUNING_COL_RENAMES = {'mean_train_neg_mean_absolute_error': 'Train MAE',
                      'mean_train_neg_mean_squared_error': 'Train MSE',
                      'mean_train_r2': 'Train R^2',
                      'mean_test_neg_mean_absolute_error': 'Test MAE',
                      'mean_test_neg_mean_squared_error': 'Test MSE',
                      'mean_test_r2': 'Test R^2',
                      'param_kneighborsregressor__n_neighbors': 'K',
                      'param_svr__C': 'C',
                      'param_xgbregressor__max_depth': 'Max Depth'}

FLIP_SIGN_RENAMED_COLS = ['Train MAE', 'Train MSE', 'Test MAE', 'Test MSE']

TUNING_TABLE_REGRESSOR_RENAMES = {'XGB': 'XGB', 'KNN': 'KNN', 'SVR': 'SVM'}


def make_test_table() -> None:
    TEST_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(TEST_RESULTS_FOLDER, TEST_RESULTS_FILENAME))
    df = df.replace({**REGRESSOR_RENAMES, **DUMMY_REGRESSOR_RENAMES})
    df = df.rename(columns=TEST_COL_RENAMES)
    df = df.round(3)
    df.to_latex(Path(TEST_RESULTS_FOLDER, 'test_error.tex'), index=False)


def make_tuning_table() -> None:
    TUNING_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    for name in TUNING_TABLE_REGRESSOR_RENAMES.keys():
        df = pd.read_csv(Path(TUNING_RESULTS_FOLDER, '{}.csv'.format(name)))
        df = df.rename(columns=TUNING_COL_RENAMES)
        df = df.drop(columns=df.columns.difference(TUNING_COL_RENAMES.values()))
        df = df.round(3)
        for col in FLIP_SIGN_RENAMED_COLS:
            df[col] = (df[col] * -1) + 0
        df.to_latex(Path(TUNING_RESULTS_FOLDER,
                         'tuning_error_{}.tex'.format(TUNING_TABLE_REGRESSOR_RENAMES[name])),
                    index=False)


def main():
    make_test_table()
    make_tuning_table()


if __name__ == '__main__':
    main()
