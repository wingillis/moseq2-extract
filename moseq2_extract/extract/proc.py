"""
Video pre-processing utilities for detecting ROIs and extracting raw data.
"""

import cv2
import joblib
import scipy.stats
import numpy as np
import scipy.signal
import skimage.measure
import scipy.interpolate
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from moseq2_extract.extract.roi import plane_ransac
from moseq2_extract.io.image import read_tiff, write_tiff
from moseq2_extract.util import convert_pxs_to_mm, strided_app
from moseq2_extract.helpers.parameters import MouseProcessing, ArenaParams
from moseq2_extract.io.video import get_movie_info, indexed_video_sequence


def get_flips(frames, flip_pipeline: Pipeline | None = None, smoothing=None):
    """
    Predict frames where mouse orientation is flipped to later correct.

    Args:
    frames (numpy.ndarray): frames x rows x columns, cropped mouse
    flip_file (str): path to pre-trained scipy random forest classifier
    smoothing (int): kernel size for median filter smoothing of random forest probabilities

    Returns:
    flips (numpy.array):  array for flips
    """

    if flip_pipeline is not None:
        flip_class = np.where(flip_pipeline.classes_ == 1)[0]

    try:
        probas = flip_pipeline.predict_proba(frames)
    except ValueError:
        if hasattr(flip_pipeline, "n_features_") and int(np.sqrt(flip_pipeline.n_features_)) != frames.shape[-1]:
            print('WARNING: Input crop-size is not compatible with flip classifier.')
            accepted_crop = int(np.sqrt(flip_pipeline.n_features_))
            print(f'Adjust the crop-size to ({accepted_crop}, {accepted_crop}) to use this flip classifier.')
        print("Frames shape:", frames.shape)
        print('The extracted data will NOT be flipped!')
        probas = np.array([[0]*len(frames), [1]*len(frames)]).T # default output; indicating no flips

    if smoothing:
        for i in range(probas.shape[1]):
            probas[:, i] = scipy.signal.medfilt(probas[:, i], smoothing)

    if flip_pipeline is not None:
        flips = probas.argmax(axis=1) == flip_class
    else:
        flips = np.zeros(len(frames), dtype=bool)

    return flips


def get_largest_cc(frames, progress_bar=False):
    """
    Returns largest connected component blob in image

    Args:
    frames (numpy.ndarray): frames x rows x columns, uncropped mouse
    progress_bar (bool): display progress bar

    Returns:
    foreground_obj (numpy.ndarray):  frames x rows x columns, true where blob was found
    """

    foreground_obj = np.zeros((frames.shape), 'bool')

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Computing largest Connected Component'):
        nb_components, output, stats, centroids =\
            cv2.connectedComponentsWithStats(frames[i], connectivity=4)
        szs = stats[:, -1]
        foreground_obj[i] = output == (szs[1:].argmax() + 1)

    return foreground_obj


def find_smoothest_background(frames: np.ndarray) -> np.ndarray:
    """Finds smoothest background to reduce the influence of a still
    mouse on background computation."""
    frames = frames.copy()
    # get rid of zeros
    frames[frames == 0] = np.nan

    smooth_outputs = {}
    bground_options = {}
    for q in np.arange(0.5, 1.0, 0.1):
        bground = np.nanquantile(frames, q, axis=0)
        gx = cv2.Sobel(bground, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(bground, cv2.CV_64F, 0, 1, ksize=5)
        gmag = cv2.magnitude(gx, gy)
        smooth_outputs[q] = np.nanmean(gmag)
        bground_options[q] = bground
    # get key for max smoothness
    q = min(smooth_outputs, key=smooth_outputs.get)
    return bground_options[q]


def get_bground_im_file(frames_file: str | Path, frame_stride=250, med_scale=5, output_dir=None, **kwargs):
    """
    Load or compute background from file.

    Args:
    frames_file (str): path to the depth video
    frame_stride (int): stride size between frames for median bground calculation
    med_scale (int): kernel size for median blur for background images.
    kwargs (dict): extra keyword arguments

    Returns:
    bground (numpy.ndarray): background image
    """
    frames_file = Path(frames_file)

    if output_dir is None:
        output_dir = frames_file.parent / 'proc'
    
    bground_path = Path(output_dir) / 'bground.tiff'
    first_frame_path = Path(output_dir) / 'first_frame.tiff'

    kwargs = deepcopy(kwargs)

    # Load background image it exists. Otherwise, compute
    if bground_path.exists() and not kwargs.get('recompute_bg', False):
        return read_tiff(bground_path, scale=True), read_tiff(first_frame_path, scale=True)

    if (finfo := kwargs.pop("finfo", None)) is None:
        finfo = get_movie_info(frames_file, **kwargs)

    finfo["dtype"] = kwargs['movie_dtype']

    frame_idx = np.arange(0, finfo['nframes'], frame_stride)
    frame_store = []
    for i, frame in enumerate(indexed_video_sequence(frames_file, frame_idx, finfo=finfo)):
        if i == 0:
            first_frame = frame.copy()
        
        frame_store.append(cv2.medianBlur(frame, med_scale))
    
    frame_store = np.array(frame_store).astype('float32')

    if kwargs.get("bg_v2", False):
        # run an optimization to determine the smoothest quantile to sample from
        bground = find_smoothest_background(frame_store)
    else:
        bground = np.nanmedian(frame_store, axis=0)

    # add zeros back
    bground = np.nan_to_num(bground)

    write_tiff(bground_path, bground, scale=True)
    write_tiff(first_frame_path, first_frame, scale=True)
        
    return bground, first_frame


def get_bbox(roi):
    """
    return an array with the x and y boundaries given ROI.

    Args:
    roi (np.ndarray): ROI boolean mask to calculate bounding box.

    Returns:
    bbox (np.ndarray): Bounding Box around ROI
    """

    y, x = np.where(roi > 0)

    if len(y) == 0 or len(x) == 0:
        return None
    else:
        bbox = np.array([[y.min(), x.min()], [y.max(), x.max()]])
        return bbox

def threshold_chunk(chunk: np.ndarray, min_height: int, max_height: int):
    """
    Zero out depth values that are less than min_height and larger than
    max_height.

    Args:
    chunk (np.ndarray): Chunk of frames to threshold (nframes, width, height)
    min_height (int): Minimum depth values to include after thresholding.
    max_height (int): Maximum depth values to include after thresholding.

    Returns:
    chunk (np.ndarray): Updated frame chunk.
    """

    chunk = np.where(np.logical_or(chunk < min_height, chunk > max_height), 0, chunk)

    return chunk

def get_roi(depth_image,
            arena_params: ArenaParams,
            return_all_data=False,
            **kwargs):
    """
    Compute an ROI using RANSAC plane fitting and simple blob features.

    Args:
        depth_image (np.ndarray): Singular depth image frame.
        return_all_data (bool): If True, returns all ROI data, else, only return ROIs and computed Planes
        arena_params (ArenaParams): Arena parameters for ROI extraction. Refer to dataclass for documentation.
        kwargs (dict) Dictionary containing `bg_roi_depth_range` parameter for plane_ransac()

    Returns:
        rois (list): list of detected roi images.
        roi_plane (np.ndarray): computed ROI Plane using RANSAC.

    If return_all_data is True, also returns:
        bboxes (list): list of computed bounding boxes for each respective ROI.
        label_im (list): list of scikit-image image properties
        ranks (list): list of ROI ranks.
        shape_index (list): list of rank means.
    """

    mask = None

    if arena_params.bg_roi_gradient_filter:
        gradient_x = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      1, 0, ksize=arena_params.bg_roi_gradient_kernel))
        gradient_y = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      0, 1, ksize=arena_params.bg_roi_gradient_kernel))
        mask = np.logical_and(gradient_x < arena_params.bg_roi_gradient_threshold, gradient_y < arena_params.bg_roi_gradient_threshold)

    roi_plane, dist_ims = plane_ransac(
        depth_image, noise_tolerance=arena_params.noise_tolerance, mask=mask,
        bg_roi_depth_range=arena_params.bg_roi_depth_range)

    if arena_params.bg_roi_gradient_filter:
        dist_ims[~mask] = np.inf

    bin_im = dist_ims < arena_params.noise_tolerance

    # anything < noise_tolerance from the plane is part of it
    label_im = skimage.measure.label(bin_im)
    region_properties = skimage.measure.regionprops(label_im)

    areas = np.zeros((len(region_properties),))
    extents = np.zeros_like(areas)
    dists = np.zeros_like(extents)

    # get the max distance from the center, area and extent
    center = np.array(depth_image.shape)/2

    for i, props in enumerate(region_properties):
        areas[i] = props.area
        extents[i] = props.extent
        tmp_dists = np.sqrt(np.sum(np.square(props.coords-center), 1))
        dists[i] = tmp_dists.max()

    # rank features
    ranks = np.vstack((scipy.stats.rankdata(-areas, method='max'),
                       scipy.stats.rankdata(-extents, method='max'),
                       scipy.stats.rankdata(dists, method='max')))
    weight_array = np.array(arena_params.bg_roi_weights, 'float32')
    shape_index = np.mean(np.multiply(ranks.astype('float32'), weight_array[:, np.newaxis]), 0).argsort()

    rois = []
    bboxes = []

    # Perform image processing on each found ROI
    for shape in shape_index:
        roi = np.zeros_like(depth_image)
        roi[region_properties[shape].coords[:, 0],
            region_properties[shape].coords[:, 1]] = 1
        if arena_params.strel_dilate is not None:
            roi = cv2.dilate(roi, arena_params.strel_dilate, iterations=arena_params.dilate_iterations) # Dilate
        if arena_params.strel_erode is not None:
            roi = cv2.erode(roi, arena_params.strel_erode, iterations=arena_params.erode_iterations) # Erode
        if arena_params.bg_roi_fill_holes:
            roi = scipy.ndimage.morphology.binary_fill_holes(roi) # Fill Holes

        rois.append(roi)
        bboxes.append(get_bbox(roi))

    if return_all_data:
        return rois, roi_plane, bboxes, label_im, ranks, shape_index
    else:
        return rois, roi_plane


def apply_roi(frames, roi):
    """
    Apply ROI to data.

    Args:
    frames (np.ndarray): input frames to apply ROI.
    roi (np.ndarray): selected ROI to extract from input images.

    Returns:
    cropped_frames (np.ndarray): Frames cropped around ROI Bounding Box.
    """

    # yeah so fancy indexing slows us down by 3-5x
    cropped_frames = frames*roi
    bbox = get_bbox(roi)

    cropped_frames = cropped_frames[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]
    return cropped_frames


def im_moment_features(IM):
    """
    Use the method of moments and centralized moments to get image properties.

    Args:
    IM (numpy.ndarray): depth image

    Returns:
    features (dict): returns a dictionary with orientation, centroid, and ellipse axis length
    """

    tmp = cv2.moments(IM)
    num = 2*tmp['mu11']
    den = tmp['mu20']-tmp['mu02']

    common = np.sqrt(4*np.square(tmp['mu11'])+np.square(den))

    if tmp['m00'] == 0:
        features = {
            'orientation': np.nan,
            'centroid': np.nan,
            'axis_length': [np.nan, np.nan]}
    else:
        features = {
            'orientation': -.5*np.arctan2(num, den),
            'centroid': [tmp['m10']/tmp['m00'], tmp['m01']/tmp['m00']],
            'axis_length': [2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']+common)/tmp['m00']),
                            2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']-common)/tmp['m00'])]
        }

    return features


def clean_frames(frames, mouse_proc_params: MouseProcessing,
                 frame_dtype='uint8', progress_bar=False):
    """
    Simple temporal and/or spatial filtering, median filter and morphological opening.

    Args:
    frames (np.ndarray): Frames (frames x rows x columns) to filter.
    mouse_proc_params (MouseProcessing): Image processing parameters to extract a mouse.
    strel_tail (cv2.StructuringElement): Element for tail filtering.
    frame_dtype (str): frame encodings
    strel_min (int): minimum kernel size
    iters_min (int): minimum number of filtering iterations
    progress_bar (bool): display progress bar

    Returns:
    filtered_frames (numpy.ndarray): frames x rows x columns
    """
    filtered_frames = frames.copy().astype(frame_dtype)

    for i in tqdm(range(len(frames)), disable=not progress_bar, desc='Cleaning frames'):
        # Erode Frames
        if mouse_proc_params.cable_filter_iters is not None and mouse_proc_params.cable_filter_iters > 0:
            filtered_frames[i] = cv2.erode(filtered_frames[i], mouse_proc_params.strel_min, mouse_proc_params.cable_filter_iters)
        # Median Blur
        if mouse_proc_params.spatial_filter_size is not None and np.all(np.array(mouse_proc_params.spatial_filter_size) > 0):
            for size in mouse_proc_params.spatial_filter_size:
                filtered_frames[i] = cv2.medianBlur(filtered_frames[i], size)
        # Tail Filter
        if mouse_proc_params.tail_filter_iters is not None and mouse_proc_params.tail_filter_iters > 0:
            filtered_frames[i] = cv2.morphologyEx(filtered_frames[i], cv2.MORPH_OPEN, mouse_proc_params.strel_tail, mouse_proc_params.tail_filter_iters)

    # Temporal Median Filter
    if mouse_proc_params.temporal_filter_size is not None and np.all(np.array(mouse_proc_params.temporal_filter_size) > 0):
        for size in mouse_proc_params.temporal_filter_size:
            filtered_frames = scipy.signal.medfilt(filtered_frames, [size, 1, 1])

    return filtered_frames


def get_frame_features(frames, frame_threshold=10, mask=None,
                       mask_threshold=-30, use_cc=False, progress_bar=False):
    """
    Use image moments to compute features of the largest object in the frame

    Args:
    frames (3d np.ndarray): input frames
    frame_threshold (int): threshold in mm separating floor from mouse
    mask (optional np.ndarray): input frame mask for parts not to filter.
    mask_threshold (int): threshold to include regions into mask.
    use_cc (bool): Use connected components.
    progress_bar (bool): Display progress bar.

    Returns:
    features (dict of lists): dictionary with simple image features
    mask (3d np.ndarray): input frame mask.
    """

    nframes = frames.shape[0]

    if not (has_mask := (mask is not None and mask.size > 0)):
        # init mask if not provided
        mask = np.zeros((frames.shape), 'uint8')

    # Pack contour features into dict
    features = {
        'centroid': np.full((nframes, 2), np.nan),
        'orientation': np.full((nframes,), np.nan),
        'axis_length': np.full((nframes, 2), np.nan)
    }

    for i in tqdm(range(nframes), disable=not progress_bar, desc='Computing moments'):
        # Threshold frame to compute mask
        frame_mask = frames[i] > frame_threshold

        # Incorporate largest connected component with frame mask
        if use_cc:
            cc_mask = get_largest_cc((frames[[i]] > mask_threshold).astype('uint8')).squeeze()
            frame_mask = np.logical_and(cc_mask, frame_mask)

        # Apply mask
        if has_mask:
            frame_mask = np.logical_and(frame_mask, mask[i] > mask_threshold)
        else:
            mask[i] = frame_mask

        # Get contours in frame
        cnts, hierarchy = cv2.findContours(frame_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmp = np.array([cv2.contourArea(x) for x in cnts])

        if tmp.size == 0:
            continue

        mouse_cnt = tmp.argmax()

        # Get features from contours
        for key, value in im_moment_features(cnts[mouse_cnt]).items():
            features[key][i] = value

    return features, mask


def crop_and_rotate_frames(frames, features, crop_size=(80, 80), progress_bar=False):
    """
    Crop mouse from image and orients it such that the head is pointing right

    Args:
    frames (3d np.ndarray): frames to crop and rotate
    features (dict): dict of extracted features, found in result_00.h5 files.
    crop_size (tuple): size of cropped image.
    progress_bar (bool): Display progress bar.

    Returns:
    cropped_frames (3d np.ndarray): Crop and rotated frames.
    """

    nframes = frames.shape[0]

    # Prepare cropped frame array
    cropped_frames = np.zeros((nframes, crop_size[0], crop_size[1]), frames.dtype)

    # Get window dimensions
    win = (crop_size[0] // 2, crop_size[1] // 2 + 1)
    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Rotating'):

        if np.any(np.isnan(features['centroid'][i])):
            continue

        # Get bounded frames
        use_frame = cv2.copyMakeBorder(frames[i], *border, cv2.BORDER_CONSTANT, 0)

        # Get row and column centroids
        rr = np.arange(features['centroid'][i, 1]-win[0],
                       features['centroid'][i, 1]+win[1]).astype('int16')
        cc = np.arange(features['centroid'][i, 0]-win[0],
                       features['centroid'][i, 0]+win[1]).astype('int16')

        rr = rr+crop_size[0]
        cc = cc+crop_size[1]

        # Ensure centroids are in bounded frame
        if (np.any(rr >= use_frame.shape[0]) or np.any(rr < 1)
                or np.any(cc >= use_frame.shape[1]) or np.any(cc < 1)):
            continue

        # Rotate the frame such that the mouse is oriented facing east
        rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2),
                                          -np.rad2deg(features['orientation'][i]), 1)
        cropped_frames[i] = cv2.warpAffine(use_frame[rr[0]:rr[-1], cc[0]:cc[-1]],
                                           rot_mat, (crop_size[0], crop_size[1]))

    return cropped_frames


def compute_scalars(frames, track_features, min_height=10, max_height=100, true_depth=673.1):
    """
    Compute extracted scalars.

    Args:
    frames (np.ndarray): frames x r x c, uncropped mouse
    track_features (dict):  dictionary with tracking variables (centroid and orientation)
    min_height (float): minimum height of the mouse
    max_height (float): maximum height of the mouse
    true_depth (float): detected true depth

    Returns:
    features (dict): dictionary of scalars
    """

    nframes = frames.shape[0]

    # Pack features into dict
    features = {
        'centroid_x_px': np.zeros((nframes,), 'float32'),
        'centroid_y_px': np.zeros((nframes,), 'float32'),
        'velocity_2d_px': np.zeros((nframes,), 'float32'),
        'velocity_3d_px': np.zeros((nframes,), 'float32'),
        'width_px': np.zeros((nframes,), 'float32'),
        'length_px': np.zeros((nframes,), 'float32'),
        'area_px': np.zeros((nframes,)),
        'centroid_x_mm': np.zeros((nframes,), 'float32'),
        'centroid_y_mm': np.zeros((nframes,), 'float32'),
        'velocity_2d_mm': np.zeros((nframes,), 'float32'),
        'velocity_3d_mm': np.zeros((nframes,), 'float32'),
        'width_mm': np.zeros((nframes,), 'float32'),
        'length_mm': np.zeros((nframes,), 'float32'),
        'area_mm': np.zeros((nframes,)),
        'height_ave_mm': np.zeros((nframes,), 'float32'),
        'angle': np.zeros((nframes,), 'float32'),
        'velocity_theta': np.zeros((nframes,)),
    }

    # Get mm centroid
    centroid_mm = convert_pxs_to_mm(track_features['centroid'], true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(track_features['centroid'] + 1, true_depth=true_depth)

    # Based on the centroid of the mouse, get the mm_to_px conversion
    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)
    masked_frames = np.logical_and(frames > min_height, frames < max_height)

    features['centroid_x_px'] = track_features['centroid'][:, 0]
    features['centroid_y_px'] = track_features['centroid'][:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion

    features['width_px'] = np.min(track_features['axis_length'], axis=1)
    features['length_px'] = np.max(track_features['axis_length'], axis=1)
    features['area_px'] = np.sum(masked_frames, axis=(1, 2))

    features['width_mm'] = features['width_px'] * px_to_mm[:, 1]
    features['length_mm'] = features['length_px'] * px_to_mm[:, 0]
    features['area_mm'] = features['area_px'] * px_to_mm.mean(axis=1)

    features['angle'] = track_features['orientation']

    nmask = np.sum(masked_frames, axis=(1, 2))

    for i in range(nframes):
        if nmask[i] > 0:
            features['height_ave_mm'][i] = np.mean(
                frames[i, masked_frames[i]])

    vel_x = np.diff(np.concatenate((features['centroid_x_px'][:1], features['centroid_x_px'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_px'][:1], features['centroid_y_px'])))
    vel_z = np.diff(np.concatenate((features['height_ave_mm'][:1], features['height_ave_mm'])))

    features['velocity_2d_px'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_px'] = np.sqrt(
        np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    vel_x = np.diff(np.concatenate((features['centroid_x_mm'][:1], features['centroid_x_mm'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_mm'][:1], features['centroid_y_mm'])))

    features['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_mm'] = np.sqrt(
        np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features


def feature_hampel_filter(features, centroid_hampel_span=None, centroid_hampel_sig=3,
                          angle_hampel_span=None, angle_hampel_sig=3):
    """
    Filter computed extraction features using Hampel Filtering.

    Args:
    features (dict): dictionary of video features
    centroid_hampel_span (int): Centroid Hampel Span Filtering Kernel Size
    centroid_hampel_sig (int): Centroid Hampel Signal Filtering Kernel Size
    angle_hampel_span (int): Angle Hampel Span Filtering Kernel Size
    angle_hampel_sig (int): Angle Hampel Span Filtering Kernel Size

    Returns:
    features (dict): filtered version of input dict.
    """
    if centroid_hampel_span is not None and centroid_hampel_span > 0:
        padded_centroids = np.pad(features['centroid'],
                                  (((centroid_hampel_span // 2, centroid_hampel_span // 2)),
                                   (0, 0)),
                                  'constant', constant_values = np.nan)
        for i in range(1):
            vws = strided_app(padded_centroids[:, i], centroid_hampel_span, 1)
            med = np.nanmedian(vws, axis=1)
            mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
            vals = np.abs(features['centroid'][:, i] - med)
            fill_idx = np.where(vals > med + centroid_hampel_sig * mad)[0]
            features['centroid'][fill_idx, i] = med[fill_idx]

        padded_orientation = np.pad(features['orientation'],
                                    (angle_hampel_span // 2, angle_hampel_span // 2),
                                    'constant', constant_values = np.nan)

    if angle_hampel_span is not None and angle_hampel_span > 0:
        vws = strided_app(padded_orientation, angle_hampel_span, 1)
        med = np.nanmedian(vws, axis=1)
        mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
        vals = np.abs(features['orientation'] - med)
        fill_idx = np.where(vals > med + angle_hampel_sig * mad)[0]
        features['orientation'][fill_idx] = med[fill_idx]

    return features


def model_smoother(features, ll=None, clips=(-300, -125)):
    """
    Apply spatial feature filtering.

    Args:
    features (dict): dictionary of extraction scalar features
    ll (numpy.array): array of loglikelihoods of pixels in frame
    clips (tuple): tuple to ensure video is indexed properly

    Returns:
    features (dict): smoothed version of input features
    """

    if ll is None or clips is None or (clips[0] >= clips[1]):
        return features

    ave_ll = np.zeros((ll.shape[0], ))
    for i, ll_frame in enumerate(ll):

        max_mu = clips[1]
        min_mu = clips[0]

        smoother = np.mean(ll[i])
        smoother -= min_mu
        smoother /= (max_mu - min_mu)

        smoother = np.clip(smoother, 0, 1)
        ave_ll[i] = smoother

    for k, v in features.items():
        nans = np.isnan(v)
        ndims = len(v.shape)
        xvec = np.arange(len(v))
        if nans.any():
            if ndims == 2:
                for i in range(v.shape[1]):
                    f = scipy.interpolate.interp1d(xvec[~nans[:, i]], v[~nans[:, i], i],
                                                   kind='nearest', fill_value='extrapolate')
                    fill_vals = f(xvec[nans[:, i]])
                    features[k][xvec[nans[:, i]], i] = fill_vals
            else:
                f = scipy.interpolate.interp1d(xvec[~nans], v[~nans],
                                               kind='nearest', fill_value='extrapolate')
                fill_vals = f(xvec[nans])
                features[k][nans] = fill_vals

    for i in range(2, len(ave_ll)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i - 1] + smoother * v[i]

    for i in reversed(range(len(ave_ll) - 1)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i + 1] + smoother * v[i]

    return features
