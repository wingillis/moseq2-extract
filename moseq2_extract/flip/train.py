import cv2
import joblib
import numpy as np
from typing import Literal
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def flatten(array: np.ndarray) -> np.ndarray:
    return array.reshape(len(array), -1)


class ImageBlurTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, blur_radius: int = 11):
        self.blur_radius = blur_radius

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if X.ndim == 3:
            X = np.array([cv2.GaussianBlur(frame, (self.blur_radius, ) * 2, 0) for frame in X])
        else:
            X = cv2.GaussianBlur(X, (self.blur_radius, ) * 2, 0)
        return X


class MaskTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mask: np.ndarray):
        """Mask should be a boolean array of length (n_pixels)"""
        self.mask = mask

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.reshape(len(X), -1)
        return X[:, self.mask].astype(np.float32)


def train_classifier(
    data_path: str,
    classifier: Literal["SVM", "RF"] = "SVM",
    n_components: int = 20,
):
    """Train a classifier to predict the orientation of a mouse.
    Parameters:
        data_path (str): Path to the training data numpy file.
        classifier (str): Classifier to use. Either 'SVM' or 'RF'.
        n_components (int): Number of components to keep in PCA."""
    data = np.load(data_path)
    frames = data["frames"]
    flipped = data["flipped"]

    mask = np.quantile(flatten(frames), 0.99, axis=0) > 10
    print(f"Number of pixels used for classification: {np.sum(mask)}/{len(mask)}")

    pipeline = make_pipeline(
        ImageBlurTransformer(),
        MaskTransformer(mask=mask),
        PCA(n_components=n_components),
        StandardScaler(),
        (
            RandomForestClassifier(n_estimators=150)
            if classifier == "RF"
            else SVC(kernel="rbf", C=2, probability=True)
        ),
    )

    print("Running cross-validation")
    accuracy = cross_val_score(
        pipeline, frames, flipped, cv=KFold(n_splits=4, shuffle=True, random_state=0), n_jobs=-1
    )
    print(f"Held-out model accuracy: {accuracy.mean()}")

    print("Final fitting step")
    return pipeline.fit(frames, flipped)


def save_classifier(clf_pipeline, out_path: str):
    joblib.dump(clf_pipeline, out_path, compress=("gzip", 3))
    print(f"Classifier saved to {out_path}")
