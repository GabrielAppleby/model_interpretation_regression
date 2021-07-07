import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from config import TUNING_RESULTS_FOLDER, SAVED_MODEL_FOLDER, SCORING
from data.data_loader import get_arrays
from modeling.regressors import REGRESSORS

RANDOM_SEED = 42


def search(pipeline: Pipeline, params: Dict, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[
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
        verbose=0)
    gs_clf: GridSearchCV = gs_clf.fit(x_train, y_train)
    best_estimator: pipeline = gs_clf.best_estimator_
    return gs_clf.cv_results_, best_estimator


# @click.command()
# @click.argument('regressor', type=click.Choice(list(REGRESSORS.keys())))
def main(regressor: str) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    TUNING_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    SAVED_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    x_train, y_train = get_arrays('train')

    clf, clf_params, clf_name = REGRESSORS[regressor]
    pipeline: Pipeline = make_pipeline(clf)

    results, best_estimator = search(pipeline, clf_params, x_train, y_train)

    dump(best_estimator, Path(SAVED_MODEL_FOLDER, '{}.joblib'.format(clf_name)))
    results['name'] = clf_name
    df = pd.DataFrame(results)
    df.to_csv(Path(TUNING_RESULTS_FOLDER, '{}.csv'.format(clf_name)), index=False)


if __name__ == '__main__':
    main('LIN')
