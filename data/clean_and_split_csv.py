import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.data_config import FULL_CSV_FILE_NAME, MIN_AGE_UNIT, HAS_EXP_ACCESS, \
    NUM_SAE, NCT_ID, DATA_FOLDER

RANDOM_SEED = 42
CONDITION_MESH = 'condition_mesh_term'
INTERVENTION_MESH = 'intervention_mesh_term'
PHASE = 'phase'

MESH_FIELDS = [CONDITION_MESH, INTERVENTION_MESH]
CATEGORICAL_FIELDS = MESH_FIELDS + [PHASE]


def drop_studies_where_age_not_measured_in_years(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df[MIN_AGE_UNIT] == 'Year') | (df[MIN_AGE_UNIT] == 'Years')]
    return df


def drop_studies_with_no_duration_or_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['actual_duration'] > 0]
    df = df[df['enrollment'] > 0]
    return df


def rename_mesh_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_rename = {'mesh_term': CONDITION_MESH, 'mesh_term.1': INTERVENTION_MESH}
    df = df.rename(columns=lambda c: to_rename[c] if c in to_rename.keys() else c)
    return df


def transform_expanded_access_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df[HAS_EXP_ACCESS] == 'f', HAS_EXP_ACCESS] = 0
    df.loc[df[HAS_EXP_ACCESS] == 't', HAS_EXP_ACCESS] = 1
    df[HAS_EXP_ACCESS].astype(int)
    return df


def divide_total_saes_by_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    df[NUM_SAE] = df[NUM_SAE] / df['enrollment']
    return df


def drop_less_popular_categorical_values(df: pd.DataFrame, column) -> pd.DataFrame:
    series = df[column]
    df = df[series.isin(series.value_counts()[:40].index)]
    return df


def transform_tall_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    dummy_fields = pd.concat(
        [df[[NCT_ID]],
         pd.get_dummies(df.drop(columns=df.columns.difference(CATEGORICAL_FIELDS)),
                        columns=CATEGORICAL_FIELDS)], axis=1).groupby(NCT_ID).max().reset_index()
    df = df.drop(columns=CATEGORICAL_FIELDS).drop_duplicates()
    df = pd.merge(dummy_fields, df, on='nct_id', how='inner')
    return df


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(FULL_CSV_FILE_NAME)
    df = drop_studies_where_age_not_measured_in_years(df)
    df = drop_studies_with_no_duration_or_enrollment(df)
    df = rename_mesh_columns(df)
    for cat_field in MESH_FIELDS:
        df = drop_less_popular_categorical_values(df, cat_field)
    df = df.drop_duplicates()
    df = df.dropna()
    unique = len(df[NCT_ID].unique())
    df = transform_expanded_access_to_binary(df)
    df = transform_tall_to_wide(df)

    assert (df.shape[0] == unique)

    df = divide_total_saes_by_enrollment(df)
    y = df[NUM_SAE]
    x = df.drop(columns=[NUM_SAE, NCT_ID, MIN_AGE_UNIT])

    splits = train_test_split(x, y, test_size=.25)
    named_splits = zip(('x_train', 'x_test', 'y_train', 'y_test'), splits)

    for name, split in named_splits:
        split.to_csv(Path(DATA_FOLDER, '{}.csv'.format(name)), index=False)


if __name__ == '__main__':
    main()
