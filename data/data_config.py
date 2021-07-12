from pathlib import Path

DATA_FOLDER: Path = Path(Path(__file__).parent, 'aact')
FULL_CSV_FILE_NAME = Path(DATA_FOLDER, 'aact_aes_data.csv')

MIN_AGE_UNIT = 'minimum_age_unit'
NUM_SAE = 'number_of_sae_subjects'
SAE_BY_ENROLL = 'sae_by_enroll'
HAS_EXP_ACCESS = 'has_expanded_access'
NCT_ID = 'nct_id'
