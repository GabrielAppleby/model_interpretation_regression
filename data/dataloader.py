from pathlib import Path

import pandas as pd

DATA_FOLDER: Path = Path(Path(__file__).parent, 'aact')

SHORTENED_COLUMNS = {'number_of_arms': 'num_arms',
                     'has_expanded_access': 'has_exp_access',
                     'number_of_facilities': 'num_facil',
                     'actual_duration': 'actual_dur',
                     'months_to_report_results': 'months_to_report',
                     'minimum_age_num': 'min_age',
                     'number_of_primary_outcomes_to_measure': 'num_primary_outcomes',
                     'number_of_secondary_outcomes_to_measure': 'num_secondary_outcomes',
                     'phase_Early Phase 1': 'p_early_1',
                     'phase_Phase 1': 'p_1',
                     'phase_Phase 1/Phase 2': 'p_1/2',
                     'phase_Phase 2': 'p_2',
                     'phase_Phase 2/Phase 3': 'p_2/3',
                     'phase_Phase 3': 'p_3',
                     'phase_Phase 4': 'p_4',
                     'number_of_sae_subjects': 'sae_by_enroll_by_dur'
                     }


def get_dataframes(split: str):
    x = pd.read_csv(Path(DATA_FOLDER, 'x_{}.csv'.format(split)))
    y = pd.read_csv(Path(DATA_FOLDER, 'y_{}.csv'.format(split)))

    df = pd.concat((x, y), axis=1)
    df = df.rename(columns=SHORTENED_COLUMNS)
    return df


def get_arrays(split: str):
    x = pd.read_csv(Path(DATA_FOLDER, 'x_{}.csv'.format(split)))
    y = pd.read_csv(Path(DATA_FOLDER, 'y_{}.csv'.format(split)))
    return x.values, y.values.ravel()
