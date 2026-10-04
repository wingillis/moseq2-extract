"""
Expectation-Maximization mouse tracking utilities.
"""

import cv2
import numpy as np
import scipy.stats
import statsmodels.stats.correlation_tools as stats_tools
from tqdm.auto import tqdm
from moseq2_extract.helpers.parameters import EMTrackingModel


def em_iter(data, mean, cov, lamd=0.1, epsilon=1e-1, max_iter=25):
    """
    Use EM tracker to iteratively update the mean and covariance variables using Expectation Maximization up to the max inputted number
    of iterations.

    Args:
    data (numpy.ndarray): frame, x, y, z coordinates to use
    mean (numpy.array): dx1, current mean estimate
    cov (numpy.array): current covariance estimate
    lambd (float): constant to add to diagonal of covariance matrix
    epsilon (float): tolerance on change in likelihood to terminate iteration
    max_iter (int): maximum number of EM iterations

    Returns:
    mean (numpy.array): updated mean
    cov (numpy.array): updated covariance
    """

    prev_likelihood = 0
    ll = 0

    ndatapoints = data.shape[1]
    pxtheta_raw = np.zeros((ndatapoints,), dtype="float64")

    for i in range(max_iter):
        pxtheta_raw = scipy.stats.multivariate_normal.pdf(x=data, mean=mean, cov=cov)
        pxtheta_raw /= np.sum(pxtheta_raw)

        mean = np.sum(data.T * pxtheta_raw, axis=1)
        dx = (data - mean).T
        cov = stats_tools.cov_nearest(np.dot(dx * pxtheta_raw, dx.T) + lamd * np.eye(3))

        ll = np.sum(np.log(pxtheta_raw + 1e-300))
        delta_likelihood = ll - prev_likelihood

        if delta_likelihood >= 0 and delta_likelihood < epsilon * abs(prev_likelihood):
            break

        prev_likelihood = ll

    return mean, cov


def em_init(
    depth_frame,
    depth_floor,
    depth_ceiling,
    init_strel,
    strel_iters=1,
):
    """
    Estimate depth frame contours using OpenCV, and select the largest chosen contour to initialize a mask for EM tracking.

    Args:
    depth_frame (np.ndarray): depth frame to initialize mask with.
    depth_floor (float): distance from camera to bucket floor.
    depth_ceiling (float): max depth value.
    init_strel (np.ndarray): structuring Element to compute mask.
    strel_iters (int): number of morphological iterations.

    Returns:
    mouse_mask (numpy.ndarray): mask of depth frame.
    """

    mask = np.logical_and(depth_frame > depth_floor, depth_frame < depth_ceiling)
    mask = cv2.morphologyEx(
        mask.astype("uint8"), cv2.MORPH_OPEN, init_strel, strel_iters
    )

    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp = np.array([cv2.contourArea(x) for x in cnts])

    try:
        use_cnt = tmp.argmax()
        mouse_mask = np.zeros_like(mask)
        cv2.drawContours(mouse_mask, cnts, use_cnt, (255), cv2.FILLED)
        mouse_mask = mouse_mask > 0
    except Exception:
        mouse_mask = mask > 0

    return mouse_mask


def em_tracking(
    frames,
    raw_frames,
    depth_floor=10,
    depth_ceiling=100,
    progress_bar=True,
    init_frames=10,
    params: EMTrackingModel | None = None,
):
    """
    Naive tracker, use EM update rules to follow a 3D Gaussian around the room.

    Args:
    frames (numpy.ndarray): filtered frames.
    raw_frames (numpy.ndarray): chunk to track mouse in.
    depth_floor (float): height in mm for separating mouse from floor
    depth_ceiling (float): max height in mm for mouse from floor.
    progress_bar (bool): display progress bar.
    init_frames (int): number of frames to include in the init calulation

    Returns:
    model_parameters (dict): mean and covariance estimates for each frame
    """

    # initialize the mean and covariance

    nframes, r, c = frames.shape
    xx, yy = np.meshgrid(np.arange(frames.shape[2]), np.arange(frames.shape[1]))
    coords = np.vstack((xx.ravel(), yy.ravel()))
    xyz = np.vstack((coords, frames[0].ravel()))

    if params.tracking_model_init_mean is None or params.tracking_model_init_cov is None:
        if params.tracking_model_init == "min":
            use_frame = np.min(frames[:init_frames], axis=0)
        elif params.tracking_model_init == "med":
            use_frame = np.median(frames[:init_frames], axis=0)
        elif params.tracking_model_init == "raw":
            use_frame = frames[0]

        mouse_mask = em_init(
            use_frame,
            depth_floor=depth_floor,
            depth_ceiling=depth_ceiling,
            init_strel=params.tracking_model_init_strel,
        )
        include_pixels = mouse_mask.ravel()

        if params.tracking_model_init_mean is None:
            try:
                mean = np.mean(xyz[:, include_pixels], axis=1)
            except Exception:
                mean = np.mean(xyz, axis=1)

        if params.tracking_model_init_cov is None:
            try:
                cov = stats_tools.cov_nearest(np.cov(xyz[:, include_pixels]))
            except Exception:
                cov = np.eye(3) * 20

        if np.any(np.isnan(mean)):
            mean = np.mean(xyz, axis=1)
    else:
        mean = params.tracking_model_init_mean
        cov = params.tracking_model_init_cov

    model_parameters = {
        "mean": np.empty((nframes, 3), "float64"),
        "cov": np.empty((nframes, 3, 3), "float64"),
    }

    for k, v in model_parameters.items():
        model_parameters[k][:] = np.nan

    frames = frames.reshape(frames.shape[0], frames.shape[1] * frames.shape[2])
    pbar = tqdm(total=nframes, disable=not progress_bar, desc="Computing EM")
    i = 0
    repeat = False
    while i < nframes:

        if repeat:
            xyz = np.vstack((coords, raw_frames[i].ravel()))
        else:
            xyz = np.vstack((coords, frames[i].ravel()))

        pxtheta_im = scipy.stats.multivariate_normal.logpdf(xyz.T, mean, cov).reshape(
            (r, c)
        )

        # segment to find pixels with likely mice, only use those for updating

        # if we try to find contours and we fail, repeat with the base initialization
        # if THAT fails, go back to the unfiltered frame and repeat base initialization
        # if THAT fails, just set all the pixels to true (tracking is hopeless, get the mouse in later frames)
        if params.tracking_model_segment and not repeat:
            try:
                cnts, hierarchy = cv2.findContours(
                    (pxtheta_im > params.tracking_model_ll_threshold).astype("uint8"),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                tmp = np.array([cv2.contourArea(x) for x in cnts])
            except Exception:
                tmp = np.array([])

            if tmp.size == 0:
                repeat = True
                continue
            else:
                use_cnt = tmp.argmax()
                mask = np.zeros_like(pxtheta_im)
                cv2.drawContours(mask, cnts, use_cnt, (255), cv2.FILLED)
        elif params.tracking_model_segment and repeat:
            # basically try each step in succession, first try to get contours
            # if that fails try re-initialization, if that fails try re-initialization
            # with raw data, if that fails give up and use all of the pixels
            mask = em_init(
                frames[i],
                depth_floor=depth_floor,
                depth_ceiling=depth_ceiling,
                init_strel=params.tracking_model_init_strel,
            )
            if np.all(mask == 0):
                mask = em_init(
                    raw_frames[i],
                    depth_floor=depth_floor,
                    depth_ceiling=depth_ceiling,
                    init_strel=params.tracking_model_init_strel,
                )
                if np.all(mask == 0):
                    mask = np.ones(pxtheta_im.shape, dtype="bool")
        else:
            mask = pxtheta_im > params.tracking_model_ll_threshold

        tmp = mask.ravel() > 0
        tmp[np.logical_or(xyz[2] <= depth_floor, xyz[2] >= depth_ceiling)] = 0

        try:
            mean_update, cov_update = em_iter(
                xyz[:, tmp.astype("bool")].T,
                mean=mean,
                cov=cov,
                epsilon=0.25,
                max_iter=15,
                lamd=30,
            )
        except Exception:
            if not repeat:
                repeat = True
                continue
            else:
                mean_update = mean
                cov_update = cov

        if (np.all(mean_update == 0) or np.all(cov_update.ravel() == 0)) and not repeat:
            print("Backing off...")
            repeat = True
            continue
        elif np.all(mean_update == 0) or np.all(cov_update.ravel() == 0):
            mean_update = np.mean(xyz, axis=1)
            cov_update = np.eye(3) * 30

        # exponential smoothers for mean and covariance if
        # you want (easier to do this offline)
        # leave these set to 0 for now
        mean = (1 - params.smoothing_rho) * mean_update + params.smoothing_rho * mean
        cov = (1 - params.smoothing_rho) * cov_update + params.smoothing_rho * cov

        model_parameters["mean"][i] = mean
        model_parameters["cov"][i] = cov

        repeat = False
        i += 1
        pbar.update(1)

    pbar.close()

    return model_parameters


def em_get_ll(frames, mean, cov, progress_bar=False):
    """
    Return mouse tracking log-likelihoods for each frame given tracking parameters.

    Args:
    frames (numpy.ndarray): depth frames
    mean (numpy.array): frames x d, mean estimates
    cov (numpy.array): frames x d x d, covariance estimates
    progress_bar (bool): use a progress bar

    Returns:
    ll (numpy.ndarray): frames x rows x columns, log likelihood of all pixels in each frame
    """

    xx, yy = np.meshgrid(np.arange(frames.shape[2]), np.arange(frames.shape[1]))
    coords = np.vstack((xx.ravel(), yy.ravel()))

    nframes, r, c = frames.shape

    ll = np.zeros(frames.shape, dtype="float64")

    for i in tqdm(
        range(nframes), disable=not progress_bar, desc="Computing EM likelihoods"
    ):
        xyz = np.vstack((coords, frames[i].ravel()))
        ll[i] = scipy.stats.multivariate_normal.logpdf(xyz.T, mean[i], cov[i]).reshape(
            (r, c)
        )

    return ll
