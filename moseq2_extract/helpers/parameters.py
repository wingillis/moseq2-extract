import cv2
import warnings
import numpy as np
from typing import Literal
from dataclasses import dataclass, field


@dataclass
class ArenaParams:
    bg_roi_shape: Literal["ellipse", "rectangle"] = "ellipse"
    bg_roi_dilate: tuple[int, int] = (10, 10)
    bg_roi_index: int = 0
    bg_roi_weights: tuple = (1, 0.1, 1)
    bg_roi_depth_range: tuple[int, int] = (650, 750)
    bg_roi_gradient_filter: bool = False
    bg_roi_gradient_kernel: int = 7
    bg_roi_gradient_threshold: float = 3000
    bg_roi_fill_holes: bool = True
    bg_roi_sort_by_area: bool = False
    bg_roi_erode: tuple[int, int] = (1, 1)
    bg_v2: bool = False
    erode_iterations: int = 0
    dilate_iterations: int = 1
    noise_tolerance: float = 30
    use_plane_bground: bool = False
    # To store the structuring elements for dilation and erosion post-init
    strel_dilate: np.ndarray = field(init=False)
    strel_erode: np.ndarray = field(init=False)

    def __post_init__(self):
        strel_map = dict(ellipse=cv2.MORPH_ELLIPSE, rectangle=cv2.MORPH_RECT)

        self.strel_dilate = cv2.getStructuringElement(
            strel_map[self.bg_roi_shape], self.bg_roi_dilate
        )
        self.strel_erode = cv2.getStructuringElement(
            strel_map[self.bg_roi_shape], self.bg_roi_erode
        )


@dataclass
class MouseProcessing:
    """
    Image processing parameters used for mouse tracking and alignment.
    """

    min_height: int = 10
    max_height: int = 120

    tail_filter_shape: Literal["ellipse", "rectangle"] = "ellipse"
    tail_filter_size: tuple[int, int] = (9, 9)
    tail_filter_iters: int = 1

    cable_filter_iters: int = 0
    cable_filter_shape: Literal["ellipse", "rectangle"] = "rectangle"
    cable_filter_size: tuple[int, int] = (5, 5)

    spatial_filter_size: list[int] = field(default_factory=lambda: [3])
    temporal_filter_size: list[int] = field(default_factory=lambda: [0])

    crop_size: tuple[int, int] = (80, 80)

    use_cc: bool = True
    use_tracking_model: bool = False

    flip_classifier: str | None = None
    flip_classifier_smoothing: int = 51

    # Store structuring elements for tail and cable filters post-init
    strel_tail: np.ndarray = field(init=False)
    strel_min: np.ndarray = field(init=False)  # for cable

    def __post_init__(self):
        # Ensure spatial and temporal filter sizes are odd numbers
        orig_spatial_filter = tuple(self.spatial_filter_size)
        orig_temporal_filter = tuple(self.temporal_filter_size)

        self.spatial_filter_size = [
            size if (size % 2 == 1) or (size == 0) else size + 1 for size in self.spatial_filter_size
        ]
        self.temporal_filter_size = [
            size if (size % 2 == 1) or (size == 0) else size + 1 for size in self.temporal_filter_size
        ]
        if tuple(self.spatial_filter_size) != orig_spatial_filter:
            warnings.warn(
                f"Spatial filter size changed from {orig_spatial_filter} to {self.spatial_filter_size}. Must be odd."
            )
        if tuple(self.temporal_filter_size) != orig_temporal_filter:
            warnings.warn(
                f"Temporal filter size changed from {orig_temporal_filter} to {self.temporal_filter_size}. Must be odd."
            )
        
        # Create structuring elements for tail and cable filters
        strel_map = dict(ellipse=cv2.MORPH_ELLIPSE, rectangle=cv2.MORPH_RECT)
        self.strel_tail = cv2.getStructuringElement(
            strel_map[self.tail_filter_shape], self.tail_filter_size
        )
        self.strel_min = cv2.getStructuringElement(
            strel_map[self.cable_filter_shape], self.cable_filter_size
        )


@dataclass
class EMTrackingModel:
    tracking_model_ll_clip: float = -100
    tracking_model_ll_threshold: float = -100
    tracking_model_mask_threshold: float = -16
    tracking_model_segment: bool = True
    tracking_model_init: str = "raw"
