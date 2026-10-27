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

Classifiers trained with moseq2-app (removed from the upgraded stack) also
load: their ``FunctionTransformer`` references
``moseq2_app.flip.train.batch_apply_pca``, which is satisfied by a faithful
pure-numpy stub so no moseq2_app installation is required.  Legacy
estimator objects additionally get the attributes modern scikit-learn
expects (PCA/FunctionTransformer fields, SVC property backing fields); the
patches are inert for the numeric arrays and reproduce bit-identical
predictions (verified against the original scikit-learn 0.20.3 stack).
"""

import gzip
import importlib
import sys
import types

import joblib
import numpy as np

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


def _moseq2_app_flatten(array):
    return array.reshape(len(array), -1)


def _moseq2_app_batch_apply_pca(frames, pca, batch_size=1000):
    """Faithful mirror of ``moseq2_app.flip.train.batch_apply_pca``: apply
    ``pca.transform`` to flattened frames in batches (a memory optimization
    only -- numerically identical to a single ``pca.transform`` call).
    Defined at module level so classifiers re-pickled through the stub stay
    loadable."""
    output = []
    if len(frames) < batch_size:
        return pca.transform(_moseq2_app_flatten(frames)).astype(np.float32)
    for arr in np.array_split(frames, len(frames) // batch_size):
        output.append(pca.transform(_moseq2_app_flatten(arr)).astype(np.float32))
    return np.concatenate(output, axis=0).astype(np.float32)


def _install_moseq2_app_stub():
    """Provide a faithful ``batch_apply_pca`` for flip classifiers pickled by
    moseq2-app (which the upgraded stack intentionally does not install).

    moseq2-app's trainer wrapped its PCA in a ``FunctionTransformer`` whose
    function reference unpickles as ``moseq2_app.flip.train.batch_apply_pca``.
    The function is a pure-numpy batched ``pca.transform`` (a memory
    optimization), so a faithful reimplementation is equivalent.  A real
    moseq2_app installation, if present, is never shadowed.
    """
    if "moseq2_app.flip.train" in sys.modules:
        return
    try:
        importlib.import_module("moseq2_app.flip.train")
        return
    except ImportError:
        pass

    pkg = types.ModuleType("moseq2_app")
    sub = types.ModuleType("moseq2_app.flip")
    mod = types.ModuleType("moseq2_app.flip.train")
    mod.batch_apply_pca = _moseq2_app_batch_apply_pca
    pkg.flip = sub
    sub.train = mod
    sys.modules["moseq2_app"] = pkg
    sys.modules["moseq2_app.flip"] = sub
    sys.modules["moseq2_app.flip.train"] = mod


def _legacy_flip_unpickler():
    """Build an unpickler class that fixes legacy sklearn references."""
    import numpy as np
    from joblib.numpy_pickle import NumpyUnpickler
    from sklearn.tree._tree import NODE_DTYPE, Tree

    class _PatchedTree(Tree):
        """Tree subclass that pads old (pre-1.3) node arrays on load and
        normalizes leaf counts (pre-0.22 pickles stored raw weighted counts;
        modern trees expect probabilities in `value`)."""

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
            # normalize per-node class counts so predict_proba matches the
            # behavior of scikit-learn < 0.22 (which divided at predict time)
            values = state.get("values") if isinstance(state, dict) else (
                state[1] if isinstance(state, (tuple, list)) else None)
            if values is not None and values.ndim == 3:
                totals = values.sum(axis=2, keepdims=True)
                totals[totals == 0] = 1.0
                state = (dict(state, values=values / totals) if isinstance(state, dict)
                         else (state[0], values / totals) + tuple(state[2:]))
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


def _patch_legacy_estimator_attrs(est):
    """Set attributes modern scikit-learn expects that older pickles lack.

    Covers the estimator families found in pre-trained flip classifiers:

    - ``PCA`` pickled before 0.24 lacks ``power_iteration_normalizer`` /
      ``iterated_power``, which modern ``__sklearn_tags__`` reads; transform()
      only uses components_/mean_, so the patch values are inert.
    - ``FunctionTransformer`` pickled before 1.2 lacks ``feature_names_out``
      and before 1.4 lacks ``set_output``.
    - ``SVC`` pickled before 0.22 stores ``n_support_`` / ``probA_`` / ``probB_``
      (and ``dual_coef_`` / ``intercept_``); modern properties read
      underscore-prefixed backing fields, and class-level properties shadow
      the instance ``__dict__`` entries restored from the pickle.
    """
    name = type(est).__name__
    if name == "PCA":
        for k, v in ("power_iteration_normalizer", "LU"), ("iterated_power", "auto"):
            if not hasattr(est, k):
                setattr(est, k, v)
    elif name == "FunctionTransformer":
        for k, v in (("feature_names_out", None), ("set_output", None),
                     ("validate", False), ("feature_names_in", None)):
            if not hasattr(est, k):
                setattr(est, k, v)
    elif name in ("SVC", "NuSVC", "SVR", "NuSVR", "OneClassSVM"):
        for pub, priv in (("n_support_", "_n_support"), ("probA_", "_probA"),
                          ("probB_", "_probB"), ("dual_coef_", "_dual_coef_"),
                          ("intercept_", "_intercept_")):
            if priv not in est.__dict__ and pub in est.__dict__:
                setattr(est, priv, est.__dict__[pub])


def _patch_legacy_estimators(obj, seen=None):
    """Add the modern `estimator` template to pre-1.2 forest pickles and patch
    legacy estimator attributes (see `_patch_legacy_estimator_attrs`)."""
    import sklearn.tree

    if seen is None:
        seen = set()
    if id(obj) in seen:
        return obj
    seen.add(id(obj))
    _patch_legacy_estimator_attrs(obj)
    if hasattr(obj, "estimators_") and not hasattr(obj, "estimator"):
        obj.estimator = sklearn.tree.DecisionTreeClassifier()
    for attr in ("estimators_",):
        for sub in getattr(obj, attr, []) or []:
            _patch_legacy_estimators(sub, seen)
    for step in getattr(obj, "steps", []) or []:
        _patch_legacy_estimators(step[1] if len(step) > 1 else step, seen)
    for v in (getattr(obj, "kw_args", None) or {}).values():
        _patch_legacy_estimators(v, seen)
    return obj


def load_flip_classifier(path):
    """Load a flip classifier pickle, tolerating legacy layouts.

    Handles pre-0.22 scikit-learn module paths and node arrays (see module
    docstring) as well as classifiers pickled by moseq2-app, whose
    ``batch_apply_pca`` reference is satisfied with a faithful stub.
    """
    _install_moseq2_app_stub()
    try:
        return _patch_legacy_estimators(joblib.load(path))
    except (ModuleNotFoundError, AttributeError):
        # Estimator classes moved to private modules; alias and retry.
        install_legacy_sklearn_aliases()
        try:
            return _patch_legacy_estimators(joblib.load(path))
        except ValueError:
            pass
    except ValueError:
        pass
    # Old decision-tree node arrays need padding; re-unpickle with fixes.
    return _patch_legacy_estimators(_load_patching_legacy_trees(path))
