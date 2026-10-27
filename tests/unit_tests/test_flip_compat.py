"""Tests for legacy flip-classifier loading (helpers/compat.py).

Covers the two additions to ``load_flip_classifier``:

1. a faithful ``moseq2_app.flip.train.batch_apply_pca`` stub, so
   classifiers pickled by moseq2-app load without that package installed;
2. attribute patches for legacy estimator objects (PCA / FunctionTransformer
   fields, SVC property backing fields) that modern scikit-learn expects.

The legacy states are simulated by surgically stripping/rening the
attributes a modern fit produces, then pickling; ``load_flip_classifier``
must restore working estimators.
"""

import sys
import types

import joblib
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

from moseq2_extract.helpers.compat import (
    load_flip_classifier,
    _install_moseq2_app_stub,
    _patch_legacy_estimators,
)


@pytest.fixture
def training_frames():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 40, size=(40, 12, 12)).astype(np.uint8)
    labels = (rng.random(40) > 0.5).astype(np.uint8)
    return frames, labels


def _strip_pca_attrs(pca):
    del pca.power_iteration_normalizer
    del pca.iterated_power


def _svc_to_legacy_state(svc):
    """Rename underscore backing fields to their pre-0.22 public names,
    mimicking what unpickling an old SVC restores into __dict__."""
    for priv, pub in (("_n_support", "n_support_"), ("_probA", "probA_"),
                      ("_probB", "probB_")):
        if priv in svc.__dict__:
            svc.__dict__[pub] = svc.__dict__.pop(priv)


class _RestoreModules:
    def __init__(self, *names):
        self.names = names
        self.saved = {n: sys.modules.get(n) for n in names}

    def __enter__(self):
        for n in self.names:
            sys.modules.pop(n, None)

    def __exit__(self, *exc):
        for n, mod in self.saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


def test_moseq2_app_stub_loads_app_pickle(tmp_path, training_frames):
    """A pickle referencing moseq2_app.flip.train.batch_apply_pca loads via
    the compat stub when moseq2_app is absent."""
    frames, labels = training_frames

    with _RestoreModules("moseq2_app", "moseq2_app.flip", "moseq2_app.flip.train"):
        # build a temporary module whose function is genuinely picklable as
        # moseq2_app.flip.train.batch_apply_pca (as the real trainer's is)
        mod = types.ModuleType("moseq2_app.flip.train")
        exec(
            "import numpy as np\n\n"
            "def batch_apply_pca(frames, pca, batch_size=1000):\n"
            "    flat = frames.reshape(len(frames), -1)\n"
            "    if len(frames) < batch_size:\n"
            "        return pca.transform(flat).astype(np.float32)\n"
            "    return np.concatenate(\n"
            "        [pca.transform(a.reshape(len(a), -1)).astype(np.float32)\n"
            "         for a in np.array_split(frames, len(frames) // batch_size)]\n"
            "    ).astype(np.float32)\n",
            mod.__dict__,
        )
        pkg = types.ModuleType("moseq2_app")
        sub = types.ModuleType("moseq2_app.flip")
        sub.train = mod
        pkg.flip = sub
        sys.modules.update({"moseq2_app": pkg, "moseq2_app.flip": sub,
                            "moseq2_app.flip.train": mod})

        pca = PCA(n_components=3).fit(frames.reshape(40, -1).astype(np.float32))
        pipeline = Pipeline([
            ("pca", FunctionTransformer(mod.batch_apply_pca,
                                        kw_args={"pca": pca}, validate=False)),
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True)),
        ]).fit(frames, labels)

        path = tmp_path / "app-style.pkl"
        joblib.dump(pipeline, path)

        # simulate the moseq2_app package being gone
        with _RestoreModules("moseq2_app", "moseq2_app.flip", "moseq2_app.flip.train"):
            clf = load_flip_classifier(path)
            probas = clf.predict_proba(frames)
            assert probas.shape == (40, 2)
            assert np.allclose(probas.sum(axis=1), 1.0)


def test_legacy_pca_and_function_transformer_attrs(tmp_path, training_frames):
    """PCA/FunctionTransformer pickles missing modern attributes predict
    after load (attrs restored by the patch walker, incl. kw_args)."""
    frames, labels = training_frames
    flat = frames.reshape(40, -1).astype(np.float32)

    # fit with intact estimators first, then strip to simulate legacy pickles
    ft = FunctionTransformer(np.asarray, validate=False, kw_args={})
    pca = PCA(n_components=3)
    pipeline = Pipeline([
        ("ft", ft), ("pca", pca), ("svc", SVC(probability=True)),
    ]).fit(flat, labels)

    _strip_pca_attrs(pipeline.named_steps["pca"])
    for attr in ("feature_names_out", "set_output", "feature_names_in"):
        if attr in ft.__dict__:
            del ft.__dict__[attr]

    path = tmp_path / "legacy-pca-ft.pkl"
    joblib.dump(pipeline, path)
    joblib.dump(ft, tmp_path / "legacy-ft.pkl")

    clf = load_flip_classifier(path)
    assert clf.predict_proba(flat).shape == (40, 2)

    loaded_ft = load_flip_classifier(str(tmp_path / "legacy-ft.pkl"))
    assert loaded_ft.feature_names_out is None


def test_legacy_svc_backing_fields(tmp_path, training_frames):
    """An SVC pickled with pre-0.22 public attribute names predicts with
    probabilities after the backing-field patch."""
    frames, labels = training_frames
    flat = frames.reshape(40, -1).astype(np.float32)

    svc = SVC(probability=True).fit(flat, labels)
    expected = svc.predict_proba(flat)
    _svc_to_legacy_state(svc)

    path = tmp_path / "legacy-svc.pkl"
    joblib.dump(svc, path)

    loaded = load_flip_classifier(path)
    probas = loaded.predict_proba(flat)
    assert probas.shape == (40, 2)
    np.testing.assert_allclose(probas, expected, atol=1e-12)


def test_patch_idempotent_and_recursive_kw_args(training_frames):
    """Patching twice is safe and reaches PCA objects inside kw_args."""
    frames, _ = training_frames
    flat = frames.reshape(40, -1).astype(np.float32)
    pca = PCA(n_components=3).fit(flat)
    _strip_pca_attrs(pca)
    ft = FunctionTransformer(lambda x: x, kw_args={"pca": pca}, validate=False)

    _patch_legacy_estimators(ft)
    _patch_legacy_estimators(ft)
    assert ft.kw_args["pca"].power_iteration_normalizer == "LU"


def test_stub_not_installed_over_real_module():
    """If a real moseq2_app.flip.train is already importable, the stub must
    not shadow it."""
    import types
    real = types.ModuleType("moseq2_app.flip.train")
    real.batch_apply_pca = "real function"
    pkg = types.ModuleType("moseq2_app")
    sub = types.ModuleType("moseq2_app.flip")
    sub.train = real
    pkg.flip = sub

    with _RestoreModules("moseq2_app", "moseq2_app.flip", "moseq2_app.flip.train"):
        sys.modules["moseq2_app"] = pkg
        sys.modules["moseq2_app.flip"] = sub
        sys.modules["moseq2_app.flip.train"] = real
        _install_moseq2_app_stub()
        assert sys.modules["moseq2_app.flip.train"].batch_apply_pca == "real function"
