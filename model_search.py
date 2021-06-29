import random
from pathlib import Path
from typing import Dict

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from config import TUNING_RESULTS_FOLDER
from data.data_loader import get_arrays
from modeling.regressors import REGRESSORS
from modeling.transforms import TRANSFORMS

RANDOM_SEED = 42



def search(pipeline: Pipeline, params: Dict, x_train: np.ndarray, y_train: np.ndarray) -> Dict:
    gs_clf: GridSearchCV = GridSearchCV(
        pipeline,
        params,
        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
        cv=3,
        n_jobs=8,
        return_train_score=True,
        error_score='raise',
        refit=False,
        verbose=0)
    gs_clf: GridSearchCV = gs_clf.fit(x_train, y_train)
    return gs_clf.cv_results_


@click.command()
@click.argument('transform', type=click.Choice(list(TRANSFORMS.keys()) + ['NONE']))
@click.argument('regressor', type=click.Choice(list(REGRESSORS.keys())))
def main(transform: str, regressor: str) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    TUNING_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    x_train, y_train = get_arrays('train')
    clf, clf_params, clf_name = REGRESSORS[regressor]

    if transform != 'NONE':
        tran, tran_params, tran_name = TRANSFORMS[transform]
        name = tran_name + '_' + clf_name
        pipeline: Pipeline = make_pipeline(tran, clf)
        params: Dict = {**tran_params, **clf_params}
    else:
        pipeline: Pipeline = make_pipeline(clf)
        params = clf_params
        name = "NONE" + '_' + clf_name

    results = search(pipeline, params, x_train, y_train)
    results['name'] = name
    df = pd.DataFrame(results)
    df.to_csv(Path(TUNING_RESULTS_FOLDER, '{}.csv'.format(name)), index=False)


if __name__ == '__main__':
    main()
