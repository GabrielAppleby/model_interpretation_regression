import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from config import TUNING_RESULTS_FOLDER, SAVED_MODEL_FOLDER, SCORING, REGRESSORS, DATASETS, \
    RANDOM_SEED, NAME_TEMPLATE


def search(pipeline: Pipeline, params: Dict, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[
    Dict, Pipeline]:
    gs_clf: GridSearchCV = GridSearchCV(
        pipeline,
        params,
        scoring=SCORING,
        cv=3,
        n_jobs=8,
        return_train_score=True,
        error_score='raise',
        refit='neg_mean_squared_error',
        verbose=2)
    gs_clf: GridSearchCV = gs_clf.fit(x_train, y_train)
    best_estimator: pipeline = gs_clf.best_estimator_
    return gs_clf.cv_results_, best_estimator


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    TUNING_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    SAVED_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    for dataset_fnc, data_name in DATASETS:
        for regressor, params, reg_name in REGRESSORS:
            name = NAME_TEMPLATE.format(model=reg_name, data=data_name)
            dataset = dataset_fnc(RANDOM_SEED)
            pipeline: Pipeline = make_pipeline(MinMaxScaler(), regressor)
            results, best_estimator = search(pipeline, params, dataset.train.features,
                                             dataset.train.targets)
            dump(best_estimator, Path(SAVED_MODEL_FOLDER, name + '.joblib'))
            results['name'] = name
            df = pd.DataFrame(results)
            df.to_csv(Path(TUNING_RESULTS_FOLDER, name + '.csv'), index=False)


if __name__ == '__main__':
    main()
