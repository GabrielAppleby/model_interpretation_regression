from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import get_scorer

from config import SAVED_MODEL_FOLDER, SCORING, TEST_RESULTS_FOLDER
from data.data_loader import get_arrays


def main():
    x_test, y_test = get_arrays('test')
    SAVED_MODEL_FOLDER.mkdir(exist_ok=True, parents=True)
    TEST_RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
    saved_model_paths = SAVED_MODEL_FOLDER.glob('*.joblib')

    models: List = [load(csv) for csv in saved_model_paths]
    results = []
    for model in models:
        current_scores = {}
        preds = model.predict(x_test)
        np.savez(Path(TEST_RESULTS_FOLDER, '{}.npz'.format(str(model))),
                 preds=preds,
                 actuals=y_test)
        current_scores['name'] = str(model)
        for score_type in SCORING:
            scorer = get_scorer(score_type)._score_func
            current_scores[score_type] = scorer(y_test, preds)
        results.append(current_scores)
    df = pd.DataFrame(results)
    df.to_csv(Path(TEST_RESULTS_FOLDER, 'final_results.csv'), index=False)


if __name__ == '__main__':
    main()
