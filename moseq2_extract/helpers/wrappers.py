"""
Wrapper functions for data processing in extraction.
"""

import sys
import uuid
import h5py
import shutil
import warnings
import numpy as np
import urllib.request
from math import ceil
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from cytoolz import partial, keyfilter, dissoc
from moseq2_extract.io.image import write_tiff
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.io.video import get_movie_info, encode_depth_to_avi, batched_video_reader, encode_depth_to_avi_batch
from moseq2_extract.helpers.parameters import MouseProcessing, ArenaParams, EMTrackingModel
from moseq2_extract.util import mouse_threshold_filter, filter_warnings, read_yaml, write_yaml
from moseq2_extract.helpers.data import (
    handle_extract_metadata,
    create_extract_h5,
    build_index_dict,
    load_extraction_meta_from_h5s,
    build_manifest,
    copy_manifest_results,
    check_completion_status,
)
from moseq2_extract.util import (
    set_bground_to_plane_fit,
    recursive_find_h5s,
    clean_dict,
    get_bucket_center,
    h5_to_dict,
    detect_and_set_camera_parameters,
    get_frame_range_indices,
    SCALAR_ATTRIBUTES,
)


def copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path):
    """
    Copy user specified metadata from h5path to a yaml file.

    Args:
    input_dir (str): path to directory containing h5 files
    h5_metadata_path (str): path within h5 to desired metadata to copy to yaml.

    """

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    to_load = [
        (tmp, yml, file)
        for tmp, yml, file in zip(dicts, yamls, h5s)
        if tmp["complete"] and not tmp["skip"]
    ]

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy

    for tup in tqdm(to_load, desc="Copying data to yamls"):
        with h5py.File(tup[2], "r") as f:
            tmp = clean_dict(h5_to_dict(f, h5_metadata_path))
            tup[0]["metadata"] = dict(tmp)

        new_file = f"{Path(tup[1]).stem}_update.yaml"
        write_yaml(new_file, tup[0])

        if new_file != tup[1]:
            shutil.move(new_file, tup[1])


@filter_warnings
def generate_index_wrapper(input_dir, output_file):
    """
    Generate index file containing a summary of all extracted sessions.

    Args:
    input_dir (str): directory to search for extracted sessions.
    output_file (str): preferred name of the index file.

    Returns:
    output_file (str): path to index file (moseq2-index.yaml).
    """

    # gather the h5s and the pca scores file
    # uuids should match keys in the scores file
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    file_with_uuids = [
        (str(Path(h5).resolve()), str(Path(yml).resolve()), meta)
        for h5, yml, meta in zip(h5s, yamls, dicts)
    ]

    # Ensuring all retrieved extracted session h5s have the appropriate metadata
    # included in their results_00.h5 file
    for file in file_with_uuids:
        try:
            if "metadata" not in file[2]:
                copy_h5_metadata_to_yaml_wrapper(input_dir, file[0])
        except:
            warnings.warn(
                f"Metadata for session {file[0]} not found. \
            File may be listed with minimal/defaulted metadata in index file."
            )

    print(f"Number of sessions included in index file: {len(file_with_uuids)}")

    # Create index file in dict form
    output_dict = build_index_dict(file_with_uuids)

    # write out index yaml
    write_yaml(output_file, output_dict)

    return output_file


def aggregate_extract_results_wrapper(
    input_dir, format, output_dir, mouse_threshold=0.0
):
    """
    Aggregate results to one folder and generate index file (moseq2-index.yaml).

    Args:
    input_dir (str): path to base directory containing all session folders
    format (str): string format for metadata to use as the new aggregated filename
    output_dir (str): name of the directory to create and store all results in
    mouse_threshold (float): threshold value of mean frame depth to include session frames

    Returns:
    indexpath (str): path to generated index file including all aggregated session information.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    h5s, dicts, _ = recursive_find_h5s(input_dir)

    def not_in_output(f: Path):
        """check if the file is already in the output directory"""
        return not (output_dir / f.name).exists()

    def complete(d: dict):
        """check if the extraction was complete and skip flag not set"""
        return d["complete"] and not d["skip"]

    # only include real extracted mice with this filter func
    mtf = partial(mouse_threshold_filter, thresh=mouse_threshold)

    def filter_h5(args):
        """remove h5's that should be skipped or extraction wasn't complete"""
        _dict, _h5 = args
        return (
            complete(_dict)
            and not_in_output(_h5)
            and mtf(_h5)
            and ("sample" not in _dict)
        )

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy
    to_load = list(filter(filter_h5, zip(dicts, h5s)))

    loaded = load_extraction_meta_from_h5s(to_load)

    manifest = build_manifest(loaded, format=format)

    copy_manifest_results(manifest, output_dir)

    print("Results successfully aggregated in", output_dir)

    indexpath = generate_index_wrapper(output_dir, Path(input_dir, "moseq2-index.yaml"))

    print(f"Index file path: {indexpath}")
    return indexpath


def generate_index_from_agg_res_wrapper(input_dir):
    """
    Generate index file from aggregated results folder.

    Args:
    input_dir (str): path to aggregated results folder
    """
    input_dir = Path(input_dir)

    # setup pca path
    pca_path = input_dir.parent / "_pca" / "pca_scores.h5"
    index_data = {
        "files": [],
        "pca_path": "",
    }
    if pca_path.exists():
        # point pca_path to existing pca scores
        index_data["pca_path"] = str(pca_path)

    for p in input_dir.rglob("*.yaml"):
        temp_yaml = read_yaml(p)
        file_dict = {
            "group": "default",
            "metadata": temp_yaml["metadata"],
            "path": [str(p.with_suffix(".h5")), str(p)],
            "uuid": temp_yaml["uuid"],
        }
        index_data["files"].append(file_dict)

    # write out index yaml
    write_yaml(input_dir.parent / "moseq2-index.yaml", index_data)


def get_roi_wrapper(input_file, config_data, output_dir=None):
    """
    Compute ROI given depth file.

    Args:
    input_file (str): path to depth file.
    config_data (dict): dictionary of ROI extraction parameters.
    output_dir (str): path to desired directory to save results in.

    Returns:
    roi (numpy.ndarray): ROI image to plot in GUI
    bground_im (numpy.ndarray): Background image to plot in GUI
    first_frame (numpy.ndarray): First frame image to plot in GUI
    """
    input_file = Path(input_file)

    # create ArenaParams object
    filtered_params = keyfilter(
        lambda k: k in ArenaParams.__dataclass_fields__, config_data
    )
    arena_params = ArenaParams(**filtered_params)
    config_data = dissoc(config_data, *filtered_params.keys())

    if output_dir is None:
        output_dir = input_file.parent / "proc"
    elif len(Path(output_dir).parts) == 1 or str(Path(output_dir).parent) not in str(input_file):
        output_dir = Path(input_file).parent / output_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_data["output_dir"] = output_dir

    if config_data.get("finfo") is None:
        # when depth video is .avi, frame_size/dim is read directly from the video
        config_data["finfo"] = get_movie_info(input_file, **config_data)

    # checks camera type to set appropriate bg_roi_weights
    config_data = detect_and_set_camera_parameters(config_data, input_file)

    print("Getting background...")
    bground_im, first_frame = get_bground_im_file(input_file, **config_data)

    # readjust depth range
    if not config_data.get("manual_set_depth_range", False):
        # search for depth values between the max distance and the halfway point to the camera
        print(
            "Automatically setting depth range. To manually set range values,"
            ' set "manual_set_depth_range" to True.\n For CLI users: use the --manual-set-depth-range flag.'
        )
        print(
            "To manually set a correct --bg-roi-depth-range value, "
            "set the min and max range values to +/-50mm of the actual camera height."
        )

        cX, cY = get_bucket_center(
            bground_im, bground_im.max(), threshold=int(np.median(bground_im) / 2)
        )
        adjusted_bg_depth_range = int(bground_im[cY][cX])
        arena_params.bg_roi_depth_range = (
            adjusted_bg_depth_range - 50, adjusted_bg_depth_range + 50
        )

    print("Getting roi...")

    rois, plane = get_roi(
        bground_im,
        **config_data,
        return_all_data=False,
        arena_params=arena_params,
    )

    if arena_params.use_plane_bground:
        print("Using plane fit for background...")
        bground_im = set_bground_to_plane_fit(bground_im, plane, output_dir)

    # Sort arena masks by largest mean area
    if arena_params.bg_roi_sort_by_area:
        rois = sorted(rois, key=lambda x: np.sum(x), reverse=True)

    if arena_params.bg_roi_index > len(rois):
        warnings.warn(
            f"bg_roi_index {arena_params.bg_roi_index} is greater than number of ROIs {len(rois)}. "
            "Setting bg_roi_index to 0."
        )
        arena_params.bg_roi_index = 0

    roi = rois[arena_params.bg_roi_index]

    roi_filename = f"roi_{arena_params.bg_roi_index:02d}.tiff"
    write_tiff(output_dir / roi_filename, roi, scale=True)

    return roi, bground_im, first_frame


def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False):
    """
    Extract depth videos.

    Args:
    input_file (str): path to depth file
    output_dir (str): path to directory to save results in.
    config_data (dict): dictionary containing extraction parameters.
    num_frames (int): number of frames to extract.
    skip (bool): indicates whether to skip file if already extracted

    Returns:
    output_dir (str): path to directory containing extraction
    """
    print("Processing:", input_file)
    # get the basic metadata
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    # filter for mouse processing parameters
    filtered_params = keyfilter(
        lambda k: k in MouseProcessing.__dataclass_fields__, config_data
    )
    mouse_proc_params = MouseProcessing(**filtered_params)
    config_data = dissoc(config_data, *filtered_params.keys())

    # filter for EM tracking parameters
    filtered_params = keyfilter(
        lambda k: k in EMTrackingModel.__dataclass_fields__, config_data
    )
    em_tracking_params = EMTrackingModel(**filtered_params)
    config_data = dissoc(config_data, *filtered_params.keys())

    # ensure 'get_cmd' and 'run_cmd' are not in config_data or get_bground_im_file will fail
    config_data = dissoc(config_data, "get_cmd", "run_cmd", "extensions")

    status_dict = {
        "complete": False,
        "skip": False,
        "uuid": str(uuid.uuid4()),
        "metadata": "",
        "parameters": deepcopy(config_data),
    }

    # loads metadata dictionary and timestamp array.
    acquisition_metadata, config_data["timestamps"] = (
        handle_extract_metadata(input_file)
    )

    config_data["finfo"] = get_movie_info(input_file, **config_data)

    if config_data["finfo"]["nframes"] is None:
        config_data["finfo"]["nframes"] = len(config_data["timestamps"])

    status_dict["metadata"] = acquisition_metadata  # update status dict

    # Getting number of frames to extract
    if num_frames is None:
        nframes = int(config_data["finfo"]["nframes"])
    elif num_frames > config_data["finfo"]["nframes"]:
        warnings.warn(
            "Requested more frames than video includes, extracting whole recording..."
        )
        nframes = int(config_data["finfo"]["nframes"])
    elif isinstance(num_frames, int):
        nframes = num_frames

    # Compute total number of frames to include from an initial starting point.
    total_frames, first_frame_idx, last_frame_idx = get_frame_range_indices(
        *config_data["frame_trim"], nframes
    )

    scalars = list(SCALAR_ATTRIBUTES)

    # set up the output directory
    if output_dir is None:
        output_dir = input_file.parent / "proc"
    elif len(output_dir.parts) < 2 and output_dir.parts[0] != "/":
        output_dir = input_file.parent / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure index is int
    if isinstance(config_data["bg_roi_index"], list):
        config_data["bg_roi_index"] = config_data["bg_roi_index"][0]

    output_filename = f'results_{config_data["bg_roi_index"]:02d}'

    results_filename = output_dir / f"{output_filename}.h5"
    movie_filename = results_filename.with_suffix(".mp4")
    status_filename = results_filename.with_suffix(".yaml")

    # Check if session has already been extracted
    if check_completion_status(status_filename) and skip:
        print("Skipping...")
        return

    write_yaml(status_filename, status_dict)

    # Compute ROIs
    roi, bground_im, first_frame = get_roi_wrapper(
        input_file, config_data, output_dir=output_dir
    )

    # Debugging option: DTD has no effect on extraction results unless dilate iterations > 1
    if config_data.get("detected_true_depth", "auto") == "auto":
        config_data["true_depth"] = np.median(bground_im[roi > 0])
    else:
        config_data["true_depth"] = int(config_data["detected_true_depth"])

    print("Detected true depth:", config_data["true_depth"])

    extraction_data = {
        "bground_im": bground_im,
        "roi": roi,
        "first_frame": first_frame,
        "first_frame_idx": first_frame_idx,
        "last_frame_idx": last_frame_idx,
        "nframes": total_frames,
    }

    # farm out the batches and write to an hdf5 file
    with h5py.File(results_filename, "w") as f:
        # Write scalars, roi, acquisition metadata, etc. to h5 file
        create_extract_h5(
            **extraction_data,
            h5_file=f,
            acquisition_metadata=acquisition_metadata,
            config_data=config_data,
            status_dict=status_dict,
            scalars_attrs=SCALAR_ATTRIBUTES,
            mouse_proc_params=mouse_proc_params,
        )

        # Write crop-rotated results to h5 file and write video preview mp4 file
        process_extract_batches(
            **extraction_data,
            h5_file=f,
            input_file=input_file,
            config_data=config_data,
            scalars=scalars,
            output_mov_path=movie_filename,
            mouse_proc_params=mouse_proc_params,
            em_tracking_params=em_tracking_params,
        )

    print()

    # Compress the depth file to avi format; compresses original raw file by ~8x.
    try:
        if input_file.suffix == ".dat" and config_data["compress"]:
            convert_raw_to_avi_wrapper(
                input_file,
                output_file=input_file.with_suffix(".avi"),
                chunk_size=config_data["chunk_size"],
                fps=config_data["fps"],
                delete=False,
            )
    except AttributeError as e:
        print("Error converting raw video to avi format, continuing anyway...")
        print(e)

    status_dict["complete"] = True
    if status_dict["parameters"].get("true_depth") is None:
        # config_data.get('true_depth') is numpy.float64 and yaml.dump can't represent the object
        status_dict["parameters"]["true_depth"] = float(config_data.get("true_depth"))

    write_yaml(status_filename, status_dict)

    return output_dir


@filter_warnings
def flip_file_wrapper(config_file, output_dir, selected_flip=None):
    """
    Download and save flip classifiers.

    Args:
    config_file (str): path to config file
    output_dir (str): path to directory to save classifier in.
    selected_flip (int or str): int: index of desired flip classifier; str: path to flip file

    Returns:
    None
    """
    output_dir = Path(output_dir)

    flip_files = {
        "large mice with fibers (K2)": "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_largemicewithfiber.pkl",
        "adult male c57s (K2)": "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_c57_10to13weeks.pkl",
        "mice with Inscopix cables (K2)": "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_inscopix.pkl",
        "adult male c57s (Azure)": "https://moseq-data.s3.amazonaws.com/flip-classifier-azure-temp.pkl",
    }

    key_list = list(flip_files)

    if selected_flip is None:
        for idx, (k, v) in enumerate(flip_files.items()):
            print(f"[{idx}] {k} ---> {v}")
    else:
        selected_flip = key_list[selected_flip]

    # prompt for user selection if not already inputted
    while selected_flip is None:
        try:
            selected_flip = key_list[int(input("Enter a selection "))]
        except (ValueError, IndexError):
            print("Please enter a valid number listed above")

    output_dir.mkdir(parents=True, exist_ok=True)

    selection = flip_files[selected_flip]

    output_filename = output_dir / Path(selection).name

    urllib.request.urlretrieve(selection, output_filename)
    print("Successfully downloaded flip file to", output_filename)

    # Update the config file with the latest path to the flip classifier
    try:
        config_data = read_yaml(config_file)
        config_data["flip_classifier"] = str(output_filename)

        write_yaml(config_file, config_data)
    except Exception as e:
        print("Could not update configuration file flip classifier path")
        print("Unexpected error:", e)


def convert_raw_to_avi_wrapper(
    input_file, output_file, chunk_size, fps, delete
):
    """
    compress a raw depth file into an avi file (with depth values) that is 8x smaller.

    Args:
    input_file (str): Path to depth file to convert
    output_file (str): Path to output avi file
    chunk_size (int): Size of frame chunks to iteratively process
    fps (int): frame rate.
    delete (bool): Delete the original depth file if True.
    """
    input_file = Path(input_file)

    if output_file is None:
        output_file = input_file.with_suffix(".avi")

    vid_info = get_movie_info(input_file)

    # Encode raw depth frames to avi file
    with encode_depth_to_avi(
        output_file, fps=fps, height=vid_info["dims"][1], width=vid_info["dims"][0]
    ) as writer:
        for indices, frames in tqdm(batched_video_reader(
            input_file, n_frames=vid_info["nframes"], batch_size=chunk_size
        ), total=ceil(vid_info["nframes"] / chunk_size), desc="Encoding raw depth frames to avi file"):
            encode_depth_to_avi_batch(frames, writer)

    # Test integrity of encoded video
    for (raw_indices, raw_frames), (encoded_indices, encoded_frames) in tqdm(
        zip(
            batched_video_reader(input_file, n_frames=vid_info["nframes"], batch_size=chunk_size),
            batched_video_reader(output_file, n_frames=vid_info["nframes"], batch_size=chunk_size),
        ),
        total=ceil(vid_info["nframes"] / chunk_size),
        desc="Testing integrity of encoded video",
    ):
        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError("Raw frames and encoded frames not equal")

    print("Encoding successful")

    if delete:
        print("Deleting", input_file)
        input_file.unlink()


def copy_slice_wrapper(
    input_file, output_file, copy_slice, chunk_size, fps, delete
):
    """
    Copy a segment of an input depth recording into a new video file. Will always encode to avi format.

    Args:
    input_file (str): Path to depth file to read segment from
    output_file (str): Path to outputted video file with copied slice.
    copy_slice (2-tuple): Frame range to copy from input file.
    chunk_size (int): Size of frame chunks to iteratively process
    fps (int): Frames per second.
    delete (bool): Delete the original depth file if True.

    Returns:
    """
    input_file = Path(input_file)

    if output_file is None:
        output_file = input_file.with_suffix(".avi")
    else:
        output_file = Path(output_file)


    vid_info = get_movie_info(input_file)
    copy_slice = (copy_slice[0], min(copy_slice[1], vid_info["nframes"]))

    if output_file.exists():
        overwrite = input(
            "Press ENTER to overwrite your previous extraction, else to end the process."
        )
        if overwrite != "":
            sys.exit(0)

    with encode_depth_to_avi(
        output_file, fps=fps, height=vid_info["dims"][1], width=vid_info["dims"][0]
    ) as writer:
        for (indices, frames) in batched_video_reader(
            input_file,
            n_frames=copy_slice[1],  # this is the end frame
            batch_size=chunk_size,
            offset=copy_slice[0],
            frame_size=vid_info["dims"],
        ):
            encode_depth_to_avi_batch(frames, writer)

    # Test integrity of encoded video
    for (raw_indices, raw_frames), (encoded_indices, encoded_frames) in zip(
        batched_video_reader(input_file, n_frames=copy_slice[1], batch_size=chunk_size, offset=copy_slice[0]),
        batched_video_reader(output_file, n_frames=copy_slice[1] - copy_slice[0], batch_size=chunk_size),
    ):
        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError("Raw frames and encoded frames not equal")

    print("Encoding successful")

    if delete:
        print("Deleting", input_file)
        input_file.unlink()
