from pathlib import Path

import pandas as pd

from vis.config import DATASETS, REGRESSORS, TUNING_RESULTS_FOLDER, NAME_TEMPLATE


def main():
    for _, data_name in DATASETS:
        for _, _, reg_name in REGRESSORS:
            name = NAME_TEMPLATE.format(model=reg_name, data=data_name)
            result_file_name = Path(TUNING_RESULTS_FOLDER, name + '.csv')
            df = pd.read_csv(result_file_name)
            hiplot_dict = {'neg_mse_test': df['mean_test_neg_mean_squared_error'],
                           'neg_mse_train': df['mean_train_neg_mean_squared_error']}
            param_col_names = [col_name for col_name in df.columns if col_name.startswith('param_')]
            for col_name in param_col_names:
                hiplot_dict[col_name.split('__')[-1]] = df[col_name]
            hiplot_df = pd.DataFrame(hiplot_dict)
            hiplot_df.to_csv(
                Path(TUNING_RESULTS_FOLDER, 'hiplot_{}_{}.csv'.format(data_name, reg_name)),
                index=False)


if __name__ == '__main__':
    main()
