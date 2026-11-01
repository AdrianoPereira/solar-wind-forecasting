"""NGBoost with single LightGBM trees as base learners, fast enough for a million pairs.

NGBoost fits one tree per distribution parameter at every iteration. With its default
scikit-learn exact trees that takes about two minutes per iteration on the hybrid
training table. Here every tree reuses one LightGBM Dataset, binned once: only the
targets (the natural gradients) and the weights change between trees.
"""

import numpy as np
from ngboost import NGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array


class LightGBMTree(BaseEstimator, RegressorMixin):
    """One LightGBM regression tree, refit on new targets without rebinning the features."""

    # Binned datasets by the identity of the feature matrix; never pickled with a model.
    _datasets = {}

    def __init__(self, num_leaves=31, min_child_samples=200, reg_lambda=1.0, threads=16, seed=42):
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.threads = threads
        self.seed = seed

    def _params(self):
        return {"objective": "regression", "learning_rate": 1.0, "num_leaves": self.num_leaves,
                "min_data_in_leaf": self.min_child_samples, "lambda_l2": self.reg_lambda,
                "num_threads": self.threads, "seed": self.seed, "deterministic": True,
                "force_row_wise": True, "verbose": -1}

    def fit(self, X, y, sample_weight=None):
        import lightgbm

        cached = self._datasets.get(id(X))
        if cached is None or cached[0] is not X:
            # Keep a reference to X so its id cannot be reused by another matrix.
            dataset = lightgbm.Dataset(X, label=y, weight=sample_weight, params=self._params(),
                                       free_raw_data=False).construct()
            self._datasets.clear()
            self._datasets[id(X)] = (X, dataset)
        else:
            dataset = cached[1]
            dataset.set_label(y)
            dataset.set_weight(sample_weight)
        self.booster_ = lightgbm.train({**self._params(), "num_iterations": 1}, dataset)
        return self

    def predict(self, X):
        return self.booster_.predict(X)


class FastNGBRegressor(NGBRegressor):
    """NGBRegressor that hands the full feature matrix to the trees when nothing is sampled."""

    def sample(self, X, Y, sample_weight, params):
        if self.minibatch_frac == 1.0 and self.col_sample == 1.0:
            return np.arange(len(Y)), np.arange(X.shape[1]), X, Y, sample_weight, params
        return super().sample(X, Y, sample_weight, params)

    def staged_predict(self, X, max_iter=None):
        """Mean after each iteration, one at a time (the parent keeps every stage in memory)."""
        X = check_array(X, ensure_all_finite="allow-nan")
        params = np.ones((len(X), self.Dist.n_params)) * self.init_params
        for i, (models, s, col_idx) in enumerate(zip(self.base_models, self.scalings, self.col_idxs), 1):
            params -= self.learning_rate * s * np.array([m.predict(X[:, col_idx]) for m in models]).T
            yield self.Dist(np.copy(params.T)).predict()
            if max_iter and i == max_iter:
                break

    def fit(self, X, Y, *args, **kwargs):
        try:
            return super().fit(np.ascontiguousarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64),
                               *args, **kwargs)
        finally:
            LightGBMTree._datasets.clear()
