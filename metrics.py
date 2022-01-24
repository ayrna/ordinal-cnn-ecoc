from functools import partial
from numbers import Real
from typing import Callable
import warnings

import numpy as np
import sklearn.metrics as skm
from numpy import ndarray as arr
from scipy.stats import gmean as stgmean, kendalltau, spearmanr
from sklearn.preprocessing import label_binarize

Scorer = Callable[[arr, arr], Real]
PerClassScorer = Callable[[arr, arr], arr]


def _per_class_sensitivity(y_true: arr, y_pred: arr) -> arr:
    cm = skm.confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    true_support = cm.sum(axis=1)

    def class_sensitivity(c):
        return cm[c, c] / true_support[c]

    return np.array([class_sensitivity(c) for c in range(n_classes)])


def _per_class_accuracy(y_true: arr, y_pred: arr) -> arr:
    cm = skm.confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    n_samples = len(y_true)

    def class_accuracy(c):
        tn = np.delete(
            np.delete(cm, c, axis=0),
            c, axis=1
        ).sum()
        tp = cm[c, c]
        return (tp + tn) / n_samples

    return np.array([class_accuracy(c) for c in range(n_classes)])


def _binary_sensitivity(y_true: arr, y_pred: arr) -> Real:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    real_pos = (y_true == 1).sum()
    return tp / real_pos


def _per_class_roc(y_true: arr, y_score: arr) -> arr:
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)

    rates = [skm.roc_curve(y_true[:, i], y_score[:, i]) for i in range(len(classes))]
    aucs = [skm.auc(fpr, tpr) for fpr, tpr, _ in rates]
    return np.array(aucs)


def _gmean(a, axis=0, dtype=None):
    if 0.0 in a:
        return 0.0
    else:
        return stgmean(a, axis, dtype)


def handle_nan(f, default):
    if np.isnan(f):
        return default
    return f


def _spearmanr(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return handle_nan(spearmanr(y_true, y_pred).correlation, 0.0)


metrics = {
    'binary': {
        'labels': {
            'accuracy': skm.accuracy_score,
            'precision': partial(skm.precision_score, zero_division=0),
            'recall': partial(skm.recall_score, zero_division=0),
            'f1-score': partial(skm.f1_score, zero_division=0),
            'sensitivity': _binary_sensitivity,
        },
        'scores': {},
    },

    'multinomial': {
        'labels': {
            'accuracy': skm.accuracy_score,
            'minimum_sensitivity': (lambda yt, yp: min(_per_class_sensitivity(yt, yp))),
            'gmean_sensitivity': (lambda yt, yp: _gmean(_per_class_sensitivity(yt, yp))),
        },
        'scores': {
            'mean_roc_auc': partial(skm.roc_auc_score, multi_class='ovr'),
        },
    },

    'multinomial_per_class': {
        'labels': {
            'accuracy': _per_class_accuracy,
            'precision': partial(skm.precision_score, average=None, zero_division=0),
            'recall': partial(skm.recall_score, average=None, zero_division=0),
            'f1-score': partial(skm.f1_score, average=None, zero_division=0),
            'sensitivity': _per_class_sensitivity,
        },
        'scores': {
            'mean_roc_auc': _per_class_roc,
        },
    },

    'ordinal': {
        'labels': {
            'rmse': (lambda yt, yp: skm.mean_squared_error(yt, yp, squared=False)*(-1)),
            'weighted_kappa_quadratic': partial(skm.cohen_kappa_score, weights='quadratic'),
            'spearman_rank_correlation': _spearmanr,
        },
        'scores': {},
    }
}
