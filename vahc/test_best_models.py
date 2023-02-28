import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import get_scorer

from config import SAVED_MODEL_FOLDER, SCORING, TEST_RESULTS_FOLDER, RAW_TEST_PREDS_FILENAME, \
    TEST_RESULTS_FILENAME
from data.data_loader import get_dataframes


RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    x_test, y_test = get_dataframes('test')
    SAVED_MODEL_FOLDER.mkdir(exist_ok=True, parents=True)
    TEST_RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
    saved_model_paths = SAVED_MODEL_FOLDER.glob('*.joblib')

    models: List = [load(csv) for csv in saved_model_paths]
    results = []
    all_preds = []
    names = []
    for model in models:
        model_name = list(model.named_steps.keys())[0]
        if model_name == 'dummyregressor':
            strategy = model.named_steps[model_name].strategy
            model_name = '{}_{}'.format(model_name, strategy)
        current_scores = {}
        preds = model.predict(x_test.values)
        all_preds.append(preds.reshape(-1, 1))
        names.append(model_name)
        current_scores['name'] = model_name
        for score_type in SCORING:
            scorer = get_scorer(score_type)._score_func
            current_scores[score_type] = scorer(y_test, preds)
        results.append(current_scores)

    raw_preds_array = np.concatenate(all_preds, axis=1)
    raw_preds_df = pd.DataFrame(raw_preds_array, columns=names)
    raw_preds_and_truth_df = pd.concat((raw_preds_df, y_test, x_test), axis=1)
    raw_preds_and_truth_df.to_csv(Path(TEST_RESULTS_FOLDER, RAW_TEST_PREDS_FILENAME), index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(TEST_RESULTS_FOLDER, TEST_RESULTS_FILENAME), index=False)


if __name__ == '__main__':
    main()
