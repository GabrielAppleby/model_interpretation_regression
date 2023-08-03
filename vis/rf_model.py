from abc import ABC, abstractmethod
from typing import *
from sklearn.ensemble._forest import ForestRegressor
from sklearn.preprocessing import normalize
from skgarden.quantile.tree import DecisionTreeQuantileRegressor
from skgarden.quantile.ensemble import generate_sample_indices
from joblib import dump, load
import pandas as pd
from scipy.stats._distn_infrastructure import rv_sample, rv_generic, rv_discrete
from scipy.stats import binom
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse, optimize
from functools import partial
from itertools import chain, combinations



class QuantileForest(ForestRegressor):   # heavily modified from scikit-garden 0.1.2
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=3,
                 min_samples_leaf=2,
                 min_weight_fraction_leaf=0.0,
                 max_features='sqrt',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(QuantileForest, self).__init__(
            base_estimator=DecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
        self.oob = oob_score
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        # apply method requires X to be of dtype np.float32
        print('Training Random Forest')
        super(QuantileForest, self).fit(X, y, sample_weight=sample_weight)

        self.y_train_ = y

        def leaves_weights(est, i=None):
            if i: print(f'Estimator: {i}\n')
            y_train_leaves_ = - np.ones(shape=(len(y),))

            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, len(y))
            else:
                bootstrap_indices = np.arange(len(y))

            est_weights = np.bincount(bootstrap_indices, minlength=len(y)) if sample_weight is None \
                else np.bincount(bootstrap_indices, minlength=len(y)) * sample_weight

            y_train_leaves = est.y_train_leaves_

            y_weights_ = sparse.csr_matrix((est_weights, (y_train_leaves, list(range(len(est_weights))))))
            y_weights_ = normalize(y_weights_, norm='l1', axis=1)
            y_weights_ = np.squeeze(np.array(np.max(y_weights_, axis=0).todense()))
            y_train_leaves_[bootstrap_indices] = y_train_leaves[bootstrap_indices]

            return y_train_leaves_, y_weights_

        print("Generating prediction distributions")

        leaves_weights_ = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(leaves_weights)(est, i)
                                                                            for i, est in enumerate(self.estimators_))

        leaves, weights = zip(*leaves_weights_)
        self.y_train_leaves_ = np.array(leaves).astype(np.int32)
        self.y_weights_ = np.array(weights).astype(np.float32)

        if self.oob:
            print('Calculating OOB predictions')
            self.oob_prediction_ = self.predict(X, oob=True, agg_values=True)
        return self

    def predict(self, X: pd.DataFrame, oob=False, agg_values=True):
        """
        Predict regression distribution for X.

        Parameters
        ----------
        X : pd.DataFrame


        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        X_leaves = self.apply(X)
        y_train_leaves_ = self.y_train_leaves_
        y_weights_ = self.y_weights_
        y_train = self.y_train_

        def value_weights(x_leaves, y, y_leaves, y_weights, estimator=None, oob=False, n=None):
            if n: print(f'Estimator {n}')
            x_l = pd.DataFrame(x_leaves)
            leaves = pd.DataFrame(index=y_leaves)
            leaves['y'] = y.values if isinstance(y, pd.Series) else y
            leaves['weights'] = y_weights
            leaves.drop(-1, inplace=True)
            leaves_g = leaves.groupby(level=0).apply(lambda g: pd.Series({'y': g.y.values, 'weights': g.weights.values}))
            x_l = x_l.join(leaves_g, on=0).drop(columns=[0])
            if oob:
                bootstrap_indices = np.unique(generate_sample_indices(estimator.random_state, len(y)))
                x_l.loc[bootstrap_indices, 'y'] = [[] for _ in range(len(bootstrap_indices))]
                x_l.loc[bootstrap_indices, 'weights'] = [[] for _ in range(len(bootstrap_indices))]
            return x_l.y.values, x_l.weights.values

        print('Gathering values and weights for predictions')
        if oob:
            all_value_weights = Parallel(n_jobs=self.n_jobs,backend='threading')\
                (delayed(value_weights)(x_l, y_train, y_l, y_w, est, oob=True, n=n)
                 for n, (x_l, y_l, y_w, est) in
                 enumerate(zip(np.transpose(X_leaves), y_train_leaves_, y_weights_, self.estimators_)))
        else:
            all_value_weights = Parallel(n_jobs=self.n_jobs,backend='threading')\
                (delayed(value_weights)(x_l, y_train, y_l, y_w, n=n)
                 for n, (x_l, y_l, y_w) in enumerate(zip(np.transpose(X_leaves), y_train_leaves_, y_weights_)))

        all_values, all_weights = list(zip(*all_value_weights))
        all_values = [np.concatenate(v) for v in zip(*all_values)]
        all_weights = [np.concatenate(v) for v in zip(*all_weights)]

        print('Generating predictions')
        forecasts = dict()
        m_y_train = np.mean(y_train)
        for n, (i, v, w) in enumerate(zip(X.index, all_values, all_weights)):
            if n % 1000 == 0:
                print(f'Data {n}')
            v_p, w_p = process_value_weights(v, w, default_value=m_y_train, agg_values=agg_values)
            forecasts[i] = rv_sample(values=(v_p, w_p))
        return pd.Series(forecasts)


def process_value_weights(values: np.array, weights: np.ndarray, default_value=None,  agg_values=False):
    if len(weights) > 0:
        if agg_values:
            v_w = pd.Series(index=values, data=weights).groupby(level=0).agg('sum')
            weights = v_w.values
            values = v_w.index.values
        idx = weights > (np.nanmax(weights) / 100)
        values_f = values[idx]
        weights = weights[idx]
        weights_n = weights / np.sum(weights)
    else:
        weights_n = np.array([1.0])
        values_f = np.array([default_value])
    return values_f, weights_n


def bayesian_update(prior: rv_sample,
                    model: Callable[[Any],rv_discrete],
                    obs: list,
                    prior_param_names: Union[List[str], str],
                    params: Union[List[Dict[str, Any]], Dict[str, Any], list] = None,
                    param_names: Union[str, List[str]] = None,
                    obs_weights: list = None,
                    agg_values=True,
                    default_value=0,
                    verbose=False) -> rv_sample:
    """
    Update prior based on observations, returns posterior

    posterior (pos_k, prob_pos_k)
    prior (pr_k, prob_pr_k)

    pos_k = pr_k
    prob_pos_k ~ \prod_{o \in obs} model_prob(o; pr_k) prob_pr_k

    :param prior: sample of prior distibution
    :param prior_params: name or list of names of prior parameters (len(prior_params) = dim(prior))
    :param params: model params not specified by prior for obs (dict if same for all obs, otherwise list of dict, len(params) = len(obs))
    :param model: discrete model, prior describe model params and obs are assumed to be observations of model
    :param obs: observations wrt perform bayesian update of prior
    :return: posterior rv_sample
    """
    if len(obs) == 0:
        return prior

    if verbose:
        print(f'Obs:{len(obs)}, priors: {len(prior.xk)}')

    if params is not None and param_names is not None:
        params = dict(zip(param_names, [np.array(p).reshape([-1, 1]) for p in params])) if isinstance(param_names, list) else {param_names: np.array(params).reshape([-1, 1]) }
    elif isinstance(params, dict):
        params = {k: np.array(v).reshape([-1, 1]) for k, v in params.items()}
    elif params is None:
        params = {}
    else:
        raise ValueError(f'Params {params} not dict but {param_names} not specified')


    prior_params = dict(zip(prior_param_names, prior.xk.reshape([1, -1]))) if isinstance(prior_param_names, list) else {prior_param_names: prior.xk.reshape([1, -1])}

    obs = np.tile(np.array(obs)[:, np.newaxis], [1, len(list(prior_params.values())[0])])

    models = model(**prior_params, **params)

    new_log_w = models.logpmf(obs)

    if obs_weights is not None:
        obs_weights = np.tile(np.array(obs_weights)[:, np.newaxis], [1, len(list(prior_params.values())[0])])
        new_log_w = new_log_w * obs_weights

    new_log_w = np.sum(new_log_w, axis=0) + np.log(prior.pk)

    # the not normalized w_i are often so small that are all numerically 0
    # we use w'_i = w_i / mean(w) instead, i.e log(w') = log(w_i)  - max(log(w_i))
    new_w = np.exp(new_log_w - np.nanmax(new_log_w))

    x_pos, w_pos = process_value_weights(prior.xk, np.array(new_w), agg_values=agg_values)

    if len(w_pos) == 0:
        x_pos = [default_value]
        w_pos = [1.0]

    try:
        return rv_sample(values=(x_pos, w_pos))
    except ValueError as e:
        print(e)
        print('w', w_pos, 'w_unrenom', new_w, 'log_w', new_log_w)


class DistributioCombination:
    def __init__(self, distr: List[rv_discrete], weights: List[float], min, max):
        self.distr = distr
        self.weights = weights
        self.bracket = [min, max]

    def cdf(self, x):
        return np.sum(np.array([d.cdf(x) * w for d, w in zip(self.distr, self.weights)]), axis=0)

    def ppf(self, q):
        is_list = isinstance(q, list)
        q = q if is_list else [q]
        quantiles = []
        for q_ in q:
            quantiles.append(optimize.root_scalar(lambda x: self.cdf(x) - q_, bracket=self.bracket).root)
        return quantiles if is_list else quantiles[0]

    def isf(self, q):
        return self.ppf(1-q)

    def median(self):
        return self.ppf(q=0.5)

    def mean(self):
        return np.sum([w*d.mean() for d, w in zip(self.distr, self.weights)])


def mixture_model(model: rv_discrete, mixture: rv_sample, param_names: Union[str, List[str]] = None, **kwargs):
    def par(p): return dict(zip(param_names, p)) if isinstance(param_names, list) else {param_names: p}
    distr = [model(**par(p)) for p in mixture.xk]
    return DistributioCombination(distr=distr, weights=mixture.pk, **kwargs)



def bayesian_update_model(prior:pd.Series,
                          prior_param_names: Union[str,List[str]],
                          model: rv_discrete,
                          obs: pd.Series,
                          obs_weights: pd.Series = None,
                          params: pd.Series = None,
                          param_names: Union[str,List[str]] = None,
                          agg_values=False,
                          verbose=False,
                          ) -> pd.Series:
    """
    Perform bayesian updates of a DiscreteProbabilisticModel.prediction
    :param prior: DiscreteProbabilisticModel.prediction
    :param prior_params: name or list of names of params specified by prior given by DiscreteProbabilisticModel.prediction
    :param model: discrete model, prior describe model params and obs are assumed to be observations of model
    :param obs: observations wrt perform bayesian update of prior
    :param params: params assumed (not specified by prior) series of params dict if const over obs otherwise list of dict for each obs
    :return: posterior DiscreteProbabilisticModel.prediction
    """
    params = params if params is not None else pd.Series(index=prior.index, data=[None] * len(prior))
    prior.name = 'prior'
    obs.name = 'obs'
    pr_obs = pd.concat([prior, obs], axis=1)

    if params is not None:
        params.name = 'params'
        pr_obs = pd.concat([pr_obs, params], axis=1)

    if obs_weights is not None:
        obs_weights.name = 'obs_weights'
        pr_obs = pd.concat([pr_obs, obs_weights], axis=1)

    return pr_obs.apply(lambda s: bayesian_update(prior_param_names=prior_param_names, model=model,
                                    agg_values=agg_values, param_names=param_names, verbose=verbose, **s.to_dict()), axis=1)


def expected(predictions: pd.Series, method='mean', wrap: Callable = None):
    return predictions.apply(lambda dist: getattr(wrap(dist) if wrap else dist, method)())


def confidence(predictions: pd.Series, confidence: float, wrap: Callable = lambda x: x):
    predictions = pd.DataFrame([[wrap(posterior).ppf((1-confidence)/2), wrap(posterior).isf((1-confidence)/2)] for posterior in predictions.values],
                            index=predictions.index, columns=['lower', 'upper'])
    return predictions


def scale(predictions: pd.Series, scaling_factor: Union[int, float, pd.Series]) -> pd.Series:
    predictions.name = 'pred'

    if isinstance(scaling_factor, pd.Series):
        scaling_factor.name = 'scale'
        pred_scale = pd.concat([predictions, scaling_factor], axis=1)
        return pred_scale.apply(lambda ps: rv_sample(values=(ps.pred.xk * ps.scale, ps.pred.pk)), axis=1)
    else:
        return predictions.apply(lambda p: rv_sample(values=(p.xk * scaling_factor, p.pk)))


def poisson_obs(df, k_col, n_col):
    def _poisson_obs(k, n):
        k_n = [k // n, (k // n) + 1]
        n_n = [(n - k % n),  k % n]
        return pd.Series({k_col: k_n, n_col: n_n})
    return df.apply(lambda r: _poisson_obs(r[k_col], r[n_col]), axis=1)


def collect(data: pd.DataFrame, gr_level: str, drop_self=True, process_cols={}, process_row=None):

    if process_row is None:
        process_row = lambda x: x

    def values(r: pd.Series):
        v = list(r.values)
        if drop_self:
            L = list(combinations(v, len(v)-1))
            L.reverse()
        else:
            L = [v] * len(r)
        return pd.Series(L, index=r.index)

    return data.groupby(level=gr_level).apply(
        lambda g: pd.DataFrame(
            {c: values(g[c]).apply(process_cols[c]) if c in process_cols else values(g[c]) for c in g}
        ).apply(process_row, axis=1))

