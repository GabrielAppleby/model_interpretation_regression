from pathlib import Path
from typing import List, Tuple, Dict

import click
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from data.dataloader import get_arrays
from regressors import CLASSIFIERS
from transforms import TRANSFORMS

TUNING_RESULTS_FOLDER: Path = Path(Path(__file__).parent, 'tuning_results')


def tune_all_classifiers(classifiers: List[Tuple], transforms: List[Tuple], train_x: np.array,
                         train_y: np.array) -> \
        Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for tran, tran_params, tran_name in transforms:
        for clf, clf_params, clf_name in classifiers:  # type: ClassifierMixin, Dict, str
            name = tran_name + '_' + clf_name
            pipeline: Pipeline = make_pipeline(tran, clf)
            print({**tran_params, **clf_params}.keys())
            tune(pipeline, {**tran_params, **clf_params}, train_x, train_y, name)

    return results


def tune(pipeline: Pipeline, params: Dict, train_x: csr_matrix, train_y: np.array, name) -> None:
    gs_clf: GridSearchCV = GridSearchCV(
        pipeline,
        params,
        cv=3,
        n_jobs=-1,
        return_train_score=True,
        error_score='raise',
        refit=True,
        verbose=0)
    gs_clf: GridSearchCV = gs_clf.fit(train_x, train_y)
    df = pd.DataFrame(gs_clf.cv_results_)
    df.to_csv(Path(TUNING_RESULTS_FOLDER, '{}.csv'.format(name)))


@click.command()
@click.argument('transform', type=click.Choice(list(TRANSFORMS.keys())))
@click.argument('regressor', type=click.Choice(list(CLASSIFIERS.keys())))
def main(transform, regressor):
    x_train, y_train = get_arrays('train')
    tran, tran_params, tran_name = TRANSFORMS[transform]
    clf, clf_params, clf_name = CLASSIFIERS[regressor]
    name = tran_name + '_' + clf_name
    pipeline: Pipeline = make_pipeline(tran, clf)
    tune(pipeline, {**tran_params, **clf_params}, x_train, y_train, name)
    # tuning_results: Dict[str, Dict] = tune_all_classifiers(CLASSIFIERS, TRANSFORMS, x_train,
    #                                                        y_train)
    # pickle.dump(best_models, open("results/best_models_nonsense.pkl", "wb"))
    # graph_tune_results(tuning_results)
    # # pickle.dump(graph_tune_results, open("results/tune_results_balanced.pkl", "wb"))
    # test_results: Dict[str, float] = evaluate(best_models, test_x, test_y)
    # # pickle.dump(test_results, open("results/test_results_balanced.pkl", "wb"))
    # print(test_results)


if __name__ == '__main__':
    # x_train, y_train = get_arrays('train')
    # tune_all_classifiers(CLASSIFIERS.values(), TRANSFORMS.values(), x_train, y_train)
    main()
