import random
from pathlib import Path

import numpy as np
import pandas as pd
from config import SAVED_MODEL_FOLDER, SCORING, TEST_RESULTS_FOLDER, REGRESSORS, DATASETS, \
    NAME_TEMPLATE, RANDOM_SEED, RAW_TEST_PREDS_TEMPLATE, TEST_RESULTS_TEMPLATE
from joblib import load
from sklearn.metrics import get_scorer


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    TEST_RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

    for dataset_fnc, data_name in DATASETS.values():
        dataset = dataset_fnc(RANDOM_SEED)
        results = []
        all_preds = []
        names = []
        for _, _, reg_name in REGRESSORS.values():
            name = NAME_TEMPLATE.format(model=reg_name, data=data_name)
            model = load(Path(SAVED_MODEL_FOLDER, name + '.joblib'))
            current_scores = {}
            preds = model.predict(dataset.test.features)
            all_preds.append(preds.reshape(-1, 1))
            names.append(name)
            current_scores['name'] = name
            for score_type in SCORING:
                scorer = get_scorer(score_type)._score_func
                current_scores[score_type] = scorer(dataset.test.targets, preds)
            results.append(current_scores)

        raw_preds_array = np.concatenate(all_preds, axis=1)
        raw_preds_df = pd.DataFrame(raw_preds_array, columns=names)
        raw_preds_and_truth_df = pd.concat((raw_preds_df.reset_index(drop=True),
                                            dataset.test.targets.reset_index(drop=True),
                                            dataset.test.features.reset_index(drop=True)), axis=1)
        raw_preds_and_truth_df.to_csv(
            Path(TEST_RESULTS_FOLDER, RAW_TEST_PREDS_TEMPLATE.format(data=data_name)), index=False)
        results_df = pd.DataFrame(results)
        results_df.to_csv(Path(TEST_RESULTS_FOLDER, TEST_RESULTS_TEMPLATE.format(data=data_name)),
                          index=False)


if __name__ == '__main__':
    main()
