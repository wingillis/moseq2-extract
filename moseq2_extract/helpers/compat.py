"""Helpers for loading flip classifiers pickled by older scikit-learn versions.

Pre-trained flip classifiers in the wild (e.g. the published K2 random
forests) were pickled with scikit-learn < 0.22, which stored estimator
classes under module paths that were later made private
(``sklearn.ensemble.forest`` -> ``sklearn.ensemble._forest`` etc.).  Those
legacy paths no longer exist, so unpickling fails with a ``ModuleNotFoundError``
on modern scikit-learn even though the estimator state itself is compatible.

``load_flip_classifier`` retries the load after mapping the legacy module
names onto their modern equivalents in ``sys.modules``.  The mapping is only
installed when a plain load fails, and never overrides modules that already
exist.
"""

import importlib
import sys

import joblib

# legacy module path -> modern module path
_LEGACY_SKLEARN_MODULES = {
    "sklearn.ensemble.forest": "sklearn.ensemble._forest",
    "sklearn.ensemble.base": "sklearn.ensemble._base",
    "sklearn.ensemble.weight_boosting": "sklearn.ensemble._weight_boosting",
    "sklearn.ensemble.gradient_boosting": "sklearn.ensemble._gradient_boosting",
    "sklearn.tree.tree": "sklearn.tree._classes",
    "sklearn.svm.classes": "sklearn.svm._classes",
    "sklearn.decomposition.pca": "sklearn.decomposition._pca",
    "sklearn.decomposition.truncated_svd": "sklearn.decomposition._truncated_svd",
    "sklearn.preprocessing.data": "sklearn.preprocessing._data",
    "sklearn.preprocessing.imputation": "sklearn.impute",
    "sklearn.linear_model.base": "sklearn.linear_model._base",
    "sklearn.linear_model.logistic": "sklearn.linear_model._logistic",
    "sklearn.utils.fixes": "sklearn.utils",
    # scikit-learn's vendored joblib (pickles from sklearn < 0.21)
    "sklearn.externals.joblib": "joblib",
    "sklearn.externals.joblib.numpy_pickle": "joblib.numpy_pickle",
    "sklearn.externals.joblib.numpy_pickle_utils": "joblib.numpy_pickle_utils",
    "sklearn.externals.joblib.my_exceptions": "joblib.my_exceptions",
    "sklearn.externals.joblib.compat": "joblib.compat",
    "sklearn.externals.joblib.disk": "joblib.disk",
}


def install_legacy_sklearn_aliases():
    """Map pre-0.22 scikit-learn module names to their modern equivalents."""
    for legacy, modern in _LEGACY_SKLEARN_MODULES.items():
        if legacy in sys.modules:
            continue
        try:
            sys.modules[legacy] = importlib.import_module(modern)
        except ImportError:
            pass


def load_flip_classifier(path):
    """Load a flip classifier pickle, tolerating legacy scikit-learn modules."""
    try:
        return joblib.load(path)
    except (ModuleNotFoundError, AttributeError):
        # Fall back to aliasing legacy sklearn module paths (removed in
        # scikit-learn >= 1.0) and retry once.
        install_legacy_sklearn_aliases()
        return joblib.load(path)
