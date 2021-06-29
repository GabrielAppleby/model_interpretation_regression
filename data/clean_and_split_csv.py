import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.data_config import FULL_CSV_FILE_NAME, MIN_AGE_UNIT, HAS_EXP_ACCESS, \
    NUM_SAE, NCT_ID, DATA_FOLDER

RANDOM_SEED = 42


def drop_studies_where_age_not_measured_in_years(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df[MIN_AGE_UNIT] == 'Year') | (df[MIN_AGE_UNIT] == 'Years')]
    return df


def drop_studies_with_no_duration_or_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['actual_duration'] > 0]
    df = df[df['enrollment'] > 0]
    return df


def transform_expanded_access_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df[HAS_EXP_ACCESS] == 'f', HAS_EXP_ACCESS] = 0
    df.loc[df[HAS_EXP_ACCESS] == 't', HAS_EXP_ACCESS] = 1
    return df


def divide_total_saes_by_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    df[NUM_SAE] = df[NUM_SAE] / df['enrollment']
    return df


def ensure_no_duplicate_studies(df: pd.DataFrame) -> None:
    assert (df.shape[0] == len(df[NCT_ID].unique()))


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(FULL_CSV_FILE_NAME)
    df = drop_studies_where_age_not_measured_in_years(df)
    df = drop_studies_with_no_duration_or_enrollment(df)
    df = df.drop_duplicates()
    df = df.dropna()

    ensure_no_duplicate_studies(df)

    df = divide_total_saes_by_enrollment(df)

    y = df[NUM_SAE]
    x = df.drop(columns=[NUM_SAE, NCT_ID, MIN_AGE_UNIT])
    x = transform_expanded_access_to_binary(x)
    x = pd.get_dummies(x, columns=['phase'])

    splits = train_test_split(x, y, test_size=.25)
    named_splits = zip(('x_train', 'x_test', 'y_train', 'y_test'), splits)

    for name, split in named_splits:
        split.to_csv(Path(DATA_FOLDER, '{}.csv'.format(name)), index=False)


if __name__ == '__main__':
    main()
