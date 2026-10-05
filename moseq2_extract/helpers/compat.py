"""Helpers for loading flip classifiers pickled by older scikit-learn versions.

Pre-trained flip classifiers in the wild (e.g. the published K2 random
forests) were pickled with scikit-learn < 0.22, and modern scikit-learn
cannot load them for two reasons:

1. Estimator classes were stored under module paths that were later made
   private (``sklearn.ensemble.forest`` -> ``sklearn.ensemble._forest``),
   which now raise ``ModuleNotFoundError``.
2. Decision-tree node arrays (scikit-learn < 1.3) lack the
   ``missing_go_to_left`` field that modern trees expect, raising a
   ``ValueError`` about an incompatible node dtype.

``load_flip_classifier`` first tries a plain ``joblib.load``.  On failure it
retries with legacy module aliases installed, and finally retries with a
custom unpickler that maps legacy module names and transparently pads the
node arrays of old trees (the padding value is 0, matching old trees that
never routed missing values).  All of this only runs when a plain load
fails; classifiers pickled with modern scikit-learn are unaffected.
"""

import gzip
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


def _legacy_flip_unpickler():
    """Build an unpickler class that fixes legacy sklearn references."""
    import numpy as np
    from joblib.numpy_pickle import NumpyUnpickler
    from sklearn.tree._tree import NODE_DTYPE, Tree

    class _PatchedTree(Tree):
        """Tree subclass that pads old (pre-1.3) node arrays on load."""

        def __setstate__(self, state):
            nodes = state.get("nodes") if isinstance(state, dict) else (
                state[0] if isinstance(state, (tuple, list)) else None)
            if nodes is not None and nodes.dtype != NODE_DTYPE:
                padded = np.zeros(nodes.shape[0], dtype=NODE_DTYPE)
                for name in nodes.dtype.names:
                    padded[name] = nodes[name]
                if isinstance(state, dict):
                    state = dict(state, nodes=padded)
                else:
                    state = (padded,) + tuple(state[1:])
            super().__setstate__(state)

    class _LegacyFlipUnpickler(NumpyUnpickler):
        def find_class(self, module, name):
            if module == "sklearn.tree._tree" and name == "Tree":
                return _PatchedTree
            if module in _LEGACY_SKLEARN_MODULES:
                modern = _LEGACY_SKLEARN_MODULES[module]
                try:
                    mod = sys.modules.get(modern) or importlib.import_module(modern)
                    return getattr(mod, name)
                except (ImportError, AttributeError):
                    pass
            return super().find_class(module, name)

    return _LegacyFlipUnpickler


def _load_patching_legacy_trees(path):
    # gzip and raw pickles cover the published classifiers; anything else
    # falls back to a plain open.
    with open(path, "rb") as f:
        head = f.read(2)
    file_opener = gzip.open if head == b"\x1f\x8b" else open

    unpickler_cls = _legacy_flip_unpickler()
    with file_opener(path, "rb") as f:
        return unpickler_cls(path, f, ensure_native_byte_order=False).load()


def load_flip_classifier(path):
    """Load a flip classifier pickle, tolerating legacy scikit-learn layouts."""
    try:
        return joblib.load(path)
    except (ModuleNotFoundError, AttributeError):
        # Estimator classes moved to private modules; alias and retry.
        install_legacy_sklearn_aliases()
        try:
            return joblib.load(path)
        except ValueError:
            pass
    except ValueError:
        pass
    # Old decision-tree node arrays need padding; re-unpickle with fixes.
    return _load_patching_legacy_trees(path)
