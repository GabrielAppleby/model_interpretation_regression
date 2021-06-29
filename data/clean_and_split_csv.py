import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
DATA_FOLDER: Path = Path(Path(__file__).parent, 'aact')
CSV_FILE_NAME = Path(DATA_FOLDER, 'aact_aes_data.csv')
MIN_AGE_UNIT = 'minimum_age_unit'
TARGET_COLUMN = 'number_of_sae_subjects'
HAS_EXP_ACCESS = 'has_expanded_access'
NCT_ID = 'nct_id'


def drop_studies_where_age_not_measured_in_years(df):
    df = df[(df[MIN_AGE_UNIT] == 'Year') | (df[MIN_AGE_UNIT] == 'Years')]
    return df


def drop_studies_with_no_duration_or_enrollment(df):
    df = df[df['actual_duration'] > 0]
    df = df[df['enrollment'] > 0]
    return df


def transform_expanded_access_to_binary(df):
    df.loc[df[HAS_EXP_ACCESS] == 'f', HAS_EXP_ACCESS] = 0
    df.loc[df[HAS_EXP_ACCESS] == 't', HAS_EXP_ACCESS] = 1
    return df


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(CSV_FILE_NAME)
    df = drop_studies_where_age_not_measured_in_years(df)
    df = drop_studies_with_no_duration_or_enrollment(df)
    df = df.drop_duplicates()
    df = df.dropna()

    assert(df.shape[0] == len(df[NCT_ID].unique()))

    y = df[TARGET_COLUMN]
    x = df.drop(columns=[TARGET_COLUMN, NCT_ID, MIN_AGE_UNIT])
    x = transform_expanded_access_to_binary(x)
    x = pd.get_dummies(x, columns=['phase'])

    splits = train_test_split(x, y, random_state=RANDOM_SEED, test_size=.25)
    named_splits = zip(('x_train', 'x_test', 'y_train', 'y_test'), splits)

    for name, split in named_splits:
        split.to_csv(Path(DATA_FOLDER, '{}.csv'.format(name)), index=False)


if __name__ == '__main__':
    main()
