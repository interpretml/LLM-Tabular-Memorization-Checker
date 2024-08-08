""" Train and evluate predictive models (logistic regression / gradient boosted tree / leave-one-out evaluation, etc.) on tabular data. """

from scipy.stats import bootstrap
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from scipy.stats import uniform
from xgboost import XGBClassifier
import numpy as np


#################################################################################################
# Accuracy and AUC with bootstraped confidence intervals
#################################################################################################


def accuracy(labels, predictions, verbose=True):
    """Compute the accuracy. Also compute a 95%-confidence interval using a bootstrap method."""
    acc = metrics.accuracy_score(labels, predictions)
    res = bootstrap(
        (np.array(labels), np.array(predictions)),
        metrics.accuracy_score,
        vectorized=False,
        paired=True,
    )
    if verbose:
        print(
            f"Accuracy: {acc:.2f}, 95%-Confidence Interval: ({res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f}), Standard error: {res.standard_error:.2f}"
        )
    return acc, res


def roc_auc(labels, predictions):
    """Compute the AUC score. Also compute a 95%-confidence interval using a bootstrap method."""
    auc = metrics.roc_auc_score(labels, predictions)
    res = bootstrap(
        (np.array(labels), np.array(predictions)),
        metrics.roc_auc_score,
        vectorized=False,
        paired=True,
    )
    print(
        f"AUC: {auc:.2f}, 95%-Confidence Interval: ({res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f})"
    )
    return auc, res


#################################################################################################
# fit and evaluate models
#################################################################################################


def fit_logistic_regression_cv(
    X_train, y_train, num_splits=7, scoring="accuracy", random_state=None
):
    """Fit a logistic regression model using cross validation.

    - Scales the input data using StandardScaler
    - Uses stratified cross validation (num_splits parameter, default 7)
    - uses a L2 penalty term and randomized search to find the best hyperparameter
    """
    assert (
        y_train.astype(int) == y_train
    ).all(), f"labels must be integers. found {y_train}"
    y_train = y_train.astype(int)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    param_space = {
        "logistic__C": uniform(0.001, 10),
    }
    clf = RandomizedSearchCV(
        pipe,
        param_space,
        scoring=scoring,
        cv=StratifiedKFold(
            n_splits=num_splits, shuffle=True, random_state=random_state
        ),
        random_state=random_state,
        verbose=0,
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def fit_gbtree_cv(
    X_train, y_train, num_splits=7, scoring="accuracy", random_state=None
):
    """Fit a gradient boosted tree (xgboost) model using cross validation. Use this for small datasets.

    - Scales the input data using StandardScaler
    - Uses stratified cross validation (num_splits parameter, default 7)
    - cross validates over: L1 penalty in: [1e-8, ..., 1]
                            L2 penalty in: [1e-8, ..., 1]
                            max_depth in:  [2, 3, 4, 6]
    Uses RandomizedSearchCV find the hyperparameters.
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    assert (
        y_train.astype(int) == y_train
    ).all(), f"labels must be integers. found {y_train}"
    y_train = y_train.astype(int)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gbtree", XGBClassifier(random_state=random_state.integers(0, 2**32))),
        ]
    )
    param_space = {
        "gbtree__max_depth": [2, 3, 4, 6],
        "gbtree__reg_alpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "gbtree__reg_lambda": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    }
    clf = RandomizedSearchCV(
        pipe,
        param_space,
        scoring=scoring,
        cv=StratifiedKFold(
            n_splits=num_splits,
            shuffle=True,
            random_state=random_state.integers(0, 2**32),
        ),
        random_state=random_state.integers(0, 2**32),
        verbose=0,
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def few_shot_with_majority_vote(num_shots, num_votes):
    """ "Decorate a fit_predict function to take exacly num_shots training points and use majority over num_votes different
    training sets to make a prediction."""

    def decorator(fit_predict_fn):
        def inner(X_train, y_train, testpoint):
            votes = []
            for _ in range(num_votes):
                X_shot, _, y_shot, _ = train_test_split(
                    X_train, y_train, train_size=num_shots
                )
                vote = fit_predict_fn(X_shot, y_shot, testpoint)
                votes.append(vote)
            # return the class that received the most votes
            return np.argmax(np.bincount(votes))

        return inner

    return decorator


def loo_eval(
    X,
    y,
    fit_predict,
    X_test=None,
    shuffle=True,
    few_shot=-1,
    stratified=True,
    max_points=1000,
    random_state=None,
):
    """Evaluate a model (given as a fit_predict function) using leave-one-out cross validation.
    This means that we fit a model on all but one data point and then evaluate it on the left out data point.
    This procedure is repeated for max_points data points.

    X_train: data points
    y_train: labels or regression target
    fit_predict(X_train, y_train, X_test): function that fits a model on X_train, y_train and returns the predictions on X_test (or sends a prompt to an LLM).
    X_test: test the loo trained models not on X, but on X_test. must have the same shape as X (Default: None)
    shuffle: whether to shuffle the data in each step (Default: True)
    few_shot: whether to reduce the training data to few_shot examples.(Default: -1, that is use all data points)
    stratified: whether to use stratified sampling in the few_shot setting. (Default: True)
    max_points: the maximum number of test points. useful for very large data sets (Default: 1000)

    Returns: The predictions for all data points. The user can then compute a metric between y and this result. The ordering of the returned predictions is the same as the ordering of the data points in X.
    """
    if few_shot > 0:
        assert shuffle == True, "few_shot only makes sense if shuffle is True"
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    predictions = np.zeros_like(y)
    for idx in range(min(X.shape[0], max_points)):
        testpoint = X[idx : idx + 1, :]
        if X_test is not None:
            testpoint = X_test[idx : idx + 1, :]
        X_loo = np.delete(X, idx, axis=0)
        y_loo = np.delete(y, idx, axis=0)
        if shuffle:
            permutation = random_state.permutation(X_loo.shape[0])
            X_loo = X_loo[permutation, :]
            y_loo = y_loo[permutation]
        if (
            few_shot > 0
        ):  # optionally reduce the training data to a given number of few shot examples
            stratify = (
                y_loo if stratified else None
            )  # optionally stratify the few shot examples
            X_loo, _, y_loo, _ = train_test_split(
                X_loo,
                y_loo,
                train_size=few_shot,
                stratify=stratify,
                random_state=random_state.integers(0, 2**32),
            )
        predictions[idx] = fit_predict(X_loo, y_loo, testpoint)
    return predictions
