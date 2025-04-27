"""
General utility functions throughout the extract package.
"""
import os
import re
import cv2
import json
import h5py
import click
import tarfile
import warnings
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from datetime import datetime
from cytoolz import valmap, concat
from moseq2_extract.io.image import write_image
from ruamel.yaml.error import UnsafeLoaderWarning
from moseq2_extract.io.video import get_movie_info
from os.path import join, exists, splitext, basename, dirname


# provides a definition for each scalar recorded in h5 file
SCALAR_ATTRIBUTES = {
    'centroid_x_px': 'X centroid (pixels)',
    'centroid_y_px': 'Y centroid (pixels)',
    'velocity_2d_px': '2D velocity (pixels / frame), note that missing frames are not accounted for',
    'velocity_3d_px': '3D velocity (pixels / frame), note that missing frames are not accounted for, also height is in mm, not pixels for calculation',
    'width_px': 'Mouse width (pixels)',
    'length_px': 'Mouse length (pixels)',
    'area_px': 'Mouse area (pixels)',
    'centroid_x_mm': 'X centroid (mm)',
    'centroid_y_mm': 'Y centroid (mm)',
    'velocity_2d_mm': '2D velocity (mm / frame), note that missing frames are not accounted for',
    'velocity_3d_mm': '3D velocity (mm / frame), note that missing frames are not accounted for',
    'width_mm': 'Mouse width (mm)',
    'length_mm': 'Mouse length (mm)',
    'area_mm': 'Mouse area (mm)',
    'height_ave_mm': 'Mouse average height (mm)',
    'angle': 'Angle (radians, unwrapped)',
    'velocity_theta': 'Angular component of velocity (arctan(vel_x, vel_y))'
}


def filter_warnings(func):
    """
    Applies warnings.simplefilter() to ignore warnings when
     running the main gui functionaity in a Jupyter Notebook.
     The function will filter out: yaml.error.UnsafeLoaderWarning, FutureWarning and UserWarning.

    Args:
    func (function): function to silence enclosed warnings.

    Returns:
    apply_warning_filters (func): Returns passed function after warnings filtering is completed.
    """
    def apply_warning_filters(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnsafeLoaderWarning)
            warnings.simplefilter(action='ignore', category=FutureWarning)
            warnings.simplefilter(action='ignore', category=UserWarning)
            return func(*args, **kwargs)
    return apply_warning_filters


def set_bground_to_plane_fit(bground_im, plane, output_dir):
    """
    Replaces median-computed background image with plane fit.
    Only occurs if config_data['use_plane_bground'] == True.

    Args:
    bground_im (numpy.ndarray): Background image computed via median value in each pixel of depth video.
    plane (numpy.ndarray): Computed ROI Plane using RANSAC.
    output_dir (str): Path to write updated background image to.

    Returns:
    bground_im (numpy.ndarray): The background image.
    """

    xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
    coords = np.vstack((xx.ravel(), yy.ravel()))

    plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
    plane_im = plane_im.reshape(bground_im.shape)

    write_image(join(output_dir, 'bground.tiff'), plane_im, scale=True)

    return plane_im

def get_frame_range_indices(trim_beginning, trim_ending, nframes):
    """
    Compute the total number of frames to be extracted, and find the start and end indices.

    Args:
    trim_beginning (int): number of frames to remove from beginning of recording
    trim_ending (int): number of frames to remove from ending of recording
    nframes (int): total number of requested frames to extract

    Returns:
    nframes (int): total number of frames to extract
    first_frame_idx (int): index of the frame to begin extraction from
    last_frame_idx (int): index of the last frame in the extraction
    """
    assert all((trim_ending >= 0, trim_beginning >= 0)) , "frame_trim arguments must be greater than or equal to 0!"

    first_frame_idx = 0
    if trim_beginning > 0 and trim_beginning < nframes:
        first_frame_idx = trim_beginning

    last_frame_idx = nframes
    if first_frame_idx < (nframes - trim_ending) and trim_ending > 0:
        last_frame_idx = nframes - trim_ending

    total_frames = last_frame_idx - first_frame_idx

    return total_frames, first_frame_idx, last_frame_idx

def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    """
    Generates batches used to chunk videos prior to extraction.

    Args:
    nframes (int): total number of frames
    chunk_size (int): the number of desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns:
    out (list): the list of batches
    """

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out

def load_timestamps(timestamp_file, col=0, alternate=False):
    """
    Read timestamps from space delimited text file for timestamps.

    Args:
    timestamp_file (str): path to timestamp file
    col (int): column in ts file read.
    alternate (boolean): specified if timestamps were saved in a csv file. False means txt file and True means csv file.

    Returns:
    ts (1D array): list of timestamps
    """

    ts = []
    try:
        with open(timestamp_file, 'r') as f:
            for line in f:
                cols = line.split()
                ts.append(float(cols[col]))
        ts = np.array(ts)
    except TypeError as e:
        # try iterating directly
        for line in timestamp_file:
            cols = line.split()
            ts.append(float(cols[col]))
        ts = np.array(ts)
    except FileNotFoundError as e:
        ts = None
        warnings.warn('Timestamp file was not found! Make sure the timestamp file exists is named '
            '"depth_ts.txt" or "timestamps.csv".')
        warnings.warn('This could cause issues for large number of dropped frames during the PCA step while '
            'imputing missing data.')

    # if timestamps were saved in a csv file
    if alternate:
        ts = ts * 1000

    return ts

def detect_avi_file(finfo):
    """
    Detect the camera type by comparing the read video resolution with known
     outputted dimensions of different camera types.

    Args:
    finfo (dict): dictionary containing the file metadata,

    Returns:
    detected (str): name of the detected camera type.
    """

    detected = 'azure'
    potential_camera_dims = {
        'kinect': [[512, 424]],
        'realsense': [[640, 480]],
        'azure': [[640, 576],
                  [320, 288],
                  [512, 512],
                  [1024, 1024]]
    }

    # Check dimensions
    if finfo is not None:
        if list(finfo['dims']) in potential_camera_dims['azure']:
            # Default Azure dimensions
            detected = 'azure'
        elif list(finfo['dims']) in potential_camera_dims['realsense']:
            # Realsense D415 output dimensions
            detected = 'realsense'
        elif list(finfo['dims']) in potential_camera_dims['kinect']:
            # Kinect output dimensions
            detected = 'kinect'
        else:
            warnings.warn('Could not infer camera type, using default Azure parameters.')

    return detected

def detect_and_set_camera_parameters(config_data, input_file=None):
    """
    Read the camera type and info and set the bg_roi_weights to the precomputed values.
    If camera_type is None, function will assume kinect is used.

    Args:
    config_data (dict): dictionary containing all input parameters.
    input_file (str): path to raw depth file

    Returns:
    config_data (dict): updated dictionary with bg-roi-weights to use for extraction.
    """

    # Auto-setting background weights
    camera_type = config_data.get('camera_type')
    finfo = config_data.get('finfo')

    default_parameters = {
        'kinect': {
            'bg_roi_weights': (1, .1, 1),
            'pixel_format': 'gray16le',
            'movie_dtype': '<u2'
        },
        'azure': {
            'bg_roi_weights': (10, 0.1, 1),
            'pixel_format': 'gray16be',
            'movie_dtype': '>u2'
        },
        'realsense': {
            'bg_roi_weights': (10, 0.1, 1),
            'pixel_format': 'gray16le',
            'movie_dtype': '<u2',
        },
    }

    if isinstance(input_file, tarfile.TarFile):
        detected = 'kinect'
    elif camera_type == 'auto' and input_file is not None:
        if input_file.endswith('.dat'):
            detected = 'kinect'
        elif input_file.endswith('.mkv'):
            detected = 'azure'
        elif input_file.endswith('.avi'):
            if finfo is None:
                finfo = get_movie_info(input_file,
                                       mapping=config_data.get('mapping', 0),
                                       threads=config_data.get('threads', 8)
                                       )
            detected = detect_avi_file(finfo)
        else:
            warnings.warn('Extension not recognized, trying default Kinect v2 parameters.')
            detected = 'kinect'

        # set the params
        config_data.update(**default_parameters[detected])
    elif camera_type in default_parameters:
        # update the config with the corresponding param set
        config_data.update(**default_parameters[camera_type])
    else:
        warnings.warn('Warning, make sure the following parameters are set to best handle your camera type: '
                      '"bg_roi_weights", "pixel_format", "movie_dtype"')

    return config_data

def check_filter_sizes(config_data):
    """
    Ensure spatial and temporal filter kernel sizes are odd numbers.

    Args:
    config_data (dict): a dictionary holding all extraction parameters

    Returns:
    config_data (dict): Updated configuration dict

    """

    # Ensure filter kernel sizes are odd
    if config_data['spatial_filter_size'][0] % 2 == 0 and config_data['spatial_filter_size'][0] > 0:
        warnings.warn("Spatial Filter Size must be an odd number. Incrementing value by 1.")
        config_data['spatial_filter_size'][0] += 1
    if config_data['temporal_filter_size'][0] % 2 == 0 and config_data['temporal_filter_size'][0] > 0:
        config_data['temporal_filter_size'][0] += 1
        warnings.warn("Spatial Filter Size must be an odd number. Incrementing value by 1.")

    return config_data

def generate_missing_metadata(sess_dir, sess_name):
    """
    Generate metadata.json with default avlues for session that does not already include one.

    Args:
    sess_dir (str): Path to session directory to create metadata.json file in.
    sess_name (str): Session Name to set the metadata SessionName.

    Returns:
    """

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': '', f'SessionName': f'{sess_name}',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}

    with open(join(sess_dir, 'metadata.json'), 'w') as fp:
        json.dump(sample_meta, fp)

def load_metadata(metadata_file):
    """
    Load metadata from session metadata.json file.

    Args:
    metadata_file (str): path to metadata file

    Returns:
    metadata (dict): metadata dictionary of JSON contents
    """

    try:
        if not exists(metadata_file):
            generate_missing_metadata(dirname(metadata_file), basename(dirname(metadata_file)))

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except TypeError:
        # try loading directly
        metadata = json.load(metadata_file)

    return metadata

def load_found_session_paths(input_dir: str | Path, exts: list[str] | str) -> list[Path]:
    """
    Find all files with the specified extension recursively in input directory.

    Args:
    input_dir (str or Path): path to project base directory holding all the session sub-folders.
    exts (list or str): list of extensions to search for, or a single extension in string form.

    Returns:
    files (list): sorted list of all paths to found files with provided extensions.
    """
    input_dir = Path(input_dir).absolute()

    if not isinstance(exts, (tuple, list)):
        exts = [exts]

    return sorted(concat(input_dir.glob('**/*' + ext) for ext in exts))

def get_strels(config_data):
    """
    Get dictionary object of cv2 StructuringElements for image filtering given
    a dict of configurations parameters.

    Args:
    config_data (dict): dict containing cv2 Structuring Element parameters

    Returns:
    str_els (dict): dict containing cv2 StructuringElements used for image filtering
    """

    str_els = {
        'strel_dilate': select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate'])),
        'strel_erode': select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_erode'])),
        'strel_tail': select_strel(config_data['tail_filter_shape'], tuple(config_data['tail_filter_size'])),
        'strel_min': select_strel(config_data['cable_filter_shape'], tuple(config_data['cable_filter_size']))
    }

    return str_els

def select_strel(string='e', size=(10, 10)):
    """
    Returns structuring element of specified shape.

    Args:
    string (str): string to indicate whether to use ellipse or rectangle
    size (tuple): size of structuring element

    Returns:
    strel (cv2.StructuringElement): selected cv2 StructuringElement to use in video filtering or ROI dilation/erosion.
    """

    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel

def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    """
    Converts x, y coordinates in pixel space to mm.

    Args:
    coords (list): list of x,y pixel coordinates
    resolution (tuple): image dimensions
    field_of_view (tuple): width and height scaling params
    true_depth (float): detected true depth

    Returns:
    new_coords (list): x,y coordinates in mm
    """

    # http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    # http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    # http://smeenk.com/kinect-field-of-view-comparison/

    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords

def convert_raw_to_avi_function(input_file, chunk_size=2000, fps=30, delete=False, threads=3):
    """
    Compress depth file (.dat, '.mkv') to avi file.

    Args:
    input_file (str): path to depth file
    chunk_size (int): size of chunks to process at a time
    fps (int): frames per second
    delete (bool): flag for deleting original depth file
    threads (int): number of threads to write video.

    """

    new_file = f'{splitext(input_file)[0]}.avi'
    print(f'Converting {input_file} to {new_file}')
    # turn into os system call...
    use_kwargs = {
        'output-file': new_file,
        'chunk-size': chunk_size,
        'fps': fps,
        'threads': threads
    }
    use_flags = {
        'delete': delete
    }
    base_command = f'moseq2-extract convert-raw-to-avi {input_file}'
    for k, v in use_kwargs.items():
        base_command += f' --{k} {v}'
    for k, v in use_flags.items():
        if v:
            base_command += f' --{k}'

    print(base_command)
    print()

    os.system(base_command)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    Create subarrays of an array with a given stride and window length.

    Args:
    a (np.ndarray) - original array
    L (int) - Window Length
    S (int) - Stride size

    Returns:
    (np.ndarray) - array of subarrays
    """

    # https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052

    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def dict_to_h5(h5, dic, root='/', annotations=None):
    """
    Save an dict to an h5 file, mounting at root.
    Keys are mapped to group names recursively.

    Args:
    h5 (h5py.File instance): h5py.file object to operate on
    dic (dict): dictionary of data to write
    root (string): group on which to add additional groups and datasets
    annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as dic.
    
    """

    if not root.endswith('/'):
        root = root + '/'

    if annotations is None:
        annotations = {} #empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in dic.items():
        dest = root + key
        try:
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5[dest] = item
            elif isinstance(item, (tuple, list)):
                h5[dest] = np.asarray(item)
            elif isinstance(item, (int, float)):
                h5[dest] = np.asarray([item])[0]
            elif item is None:
                h5.create_dataset(dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str)))
            elif isinstance(item, dict):
                dict_to_h5(h5, item, dest)
            else:
                raise ValueError('Cannot save {} type to key {}'.format(type(item), dest))
        except Exception as e:
            print(e)
            if key != 'inputs':
                print('h5py could not encode key:', key)

        if key in annotations:
            if annotations[key] is None:
                h5[dest].attrs['description'] = ""
            else:
                h5[dest].attrs['description'] = annotations[key]


def _walk_and_filter(root_dir, filter_func):
    """
    Helper to walk a directory recursively and apply a filter function to each file.

    Args:
        root_dir (str): The root directory to start walking from.
        filter_func (callable): A function that takes (root, file_name) and returns True
                                if the file should be included, False otherwise.

    Returns:
        list[str]: A list of absolute paths to the files that passed the filter.
    """
    matched_paths = []
    abs_root_dir = Path(root_dir).absolute()
    for root, _, files in os.walk(abs_root_dir):
        root = Path(root)
        for file_path in map(lambda f: root / f, files):
            if filter_func(file_path):
                matched_paths.append(file_path)
    return matched_paths


def recursive_find_h5s(root_dir=Path.cwd(),
                       ext='.h5',
                       yaml_suffix='.yaml'):
    """
    Recursively find h5 files, along with yaml files with the same basename,
    that contain a 'frames' dataset.

    Args:
    root_dir (str): path to base directory to begin recursive search in.
    ext (str): extension to search for
    yaml_suffix (str): string for filename formatting when finding related yaml files

    Returns:
    h5s (list): list of found h5 files meeting criteria
    dicts (list): list of corresponding loaded yaml file contents as dictionaries
    yamls (list): list of corresponding found yaml file paths
    """
    if not ext.startswith('.'):
        ext = '.' + ext

    def has_frames(f_path: Path):
        try:
            with h5py.File(f_path, 'r') as h5f:
                return 'frames' in h5f
        except OSError:
            warnings.warn(f'Error reading {f_path}, skipping...')
            return False
        except Exception as e:
            warnings.warn(f'Unexpected error reading {f_path}: {e}, skipping...')
            return False

    def _filter_h5(h5_path: Path):
        if h5_path.suffix != ext:
            return False

        yaml_file = h5_path.with_suffix(yaml_suffix)
        return yaml_file.exists() and has_frames(h5_path)

    # Use the helper function to find valid H5 files
    h5s_final = _walk_and_filter(root_dir, _filter_h5)

    # Generate corresponding yamls and dicts, handling potential read errors
    yamls = []
    dicts = []
    valid_h5s = []
    for h5_file in h5s_final:
        yaml_file = h5_file.with_suffix(yaml_suffix)
        try:
            dicts.append(read_yaml(yaml_file))
            yamls.append(yaml_file)
            valid_h5s.append(h5_file) # Keep h5 file only if yaml read succeeds
        except Exception as e_read:
            warnings.warn(f"Skipping H5 {h5_file} due to error reading YAML {yaml_file}: {e_read}")

    return valid_h5s, dicts, yamls


def load_textdata(data_file, dtype=np.float32):
    """
    Loads timestamp from txt/csv file.

    Args:
    data_file (str): path to timestamp file
    dtype (dtype): data type of timestamps

    Returns:
    data (np.ndarray): timestamp data
    timestamps (numpy.array): the array for the timestamps
    """

    data = []
    timestamps = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            tmp = line.split(' ', 1)
            # appending timestamp value
            timestamps.append(int(float(tmp[0])))

            # append data indicator value
            clean_data = np.fromstring(tmp[1].replace(" ", "").strip(), sep=',', dtype=dtype)
            data.append(clean_data)

    data = np.stack(data, axis=0).squeeze()
    timestamps = np.array(timestamps, dtype=np.int)

    return data, timestamps


def build_path(keys: dict, format_string: str, snake_case=True) -> str:
    """
    Produce a new file name using keys collected from extraction h5 files.

    Args:
    keys (dict): dictionary specifying which keys used to produce the new file name
    format_string (str): the string to reformat using the `keys` dictionary i.e. '{subject_name}_{session_name}'.
    snake_case (bool): flag to save the files with snake_case

    Returns:
    out (str): a newly formatted filename useable with any operating system
    """

    if 'start_time' in keys:
        # Parse the ISO‐8601 timestamp (with offset) and format for filenames
        dt = datetime.fromisoformat(keys['start_time'])
        keys['start_time'] = dt.strftime("%Y-%m-%d_%H-%M-%S")

    if snake_case:
        keys = valmap(camel_to_snake, keys)

    formatted = format_string.format(**keys)
    # remove invalid characters for a file name
    formatted = re.sub(r'[ <>:"/\\|?*\']', '-', formatted).replace('--', '-')
    return formatted


def read_yaml(yaml_file):
    """
    Read yaml file into a dictionary

    Args:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml contents
    """
    yaml = YAML(typ='safe', pure=True)

    with open(yaml_file, 'r') as f:
        return yaml.load(f)

def mouse_threshold_filter(h5file, thresh=0):
    """
    Filter frames in h5 files by threshold value.

    Args:
    h5file (str): path to h5 file
    thresh (int): threshold at which to apply filter

    Returns:
    (3d-np boolean array): array of regions to include after threshold filter.
    """

    with h5py.File(h5file, 'r') as f:
        # select 1st 1000 frames
        frames = f['frames'][:min(f['frames'].shape[0], 1000)]
    return np.nanmean(frames) > thresh

def _load_h5_to_dict(file: h5py.File, path) -> dict:
    """
    Loads h5 contents to dictionary object.

    Args:
    h5file (h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns:
    ans (dict): a dict with h5 file contents with the same path structure
    """

    ans = {}
    for key, item in file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path) -> dict:
    """
    Load h5 contents to dictionary object.

    Args:
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns:
    out (dict): a dict with h5 file contents with the same path structure
    """

    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, h5py.File):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise ValueError('file input not understood - need h5 file path or file object')
    return out

def clean_dict(dct: dict) -> dict:
    """
    Standardize types of dict value.

    Args:
    dct (dict): dict object with mixed type value objects.

    Returns:
    out (dict): dict object with list value objects.
    """

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            out = np.asscalar(e)
        else:
            out = e
        return out

    return valmap(clean_entry, dct)

_underscorer = re.compile(r'(?<!^)(?=[A-Z])')

def camel_to_snake(s: str) -> str:
    """
    Convert CamelCase to snake_case.
    """
    return _underscorer.sub('_', s).lower()

def recursive_find_unextracted_dirs(root_dir: Path = Path.cwd(),
                                    session_pattern=r'session_\d+\.(?:tgz|tar\.gz)',
                                    extension='.dat',
                                    yaml_path='proc/results_00.yaml',
                                    metadata_path='metadata.json',
                                    skip_checks=False):
    """
    Recursively find unextracted (or incompletely extracted) directories by checking
    for source data files (.dat or archives) and the status of their expected outputs.

    Args:
    root_dir (str): path to base directory to start recursive search.
    session_pattern (str): regex pattern for session archive filenames.
    extension (str): file extension for raw data files.
    yaml_path (str): relative path from session dir to the completion status yaml file.
    metadata_path (str): relative path from session dir (or root for archives) to the metadata json file.
    skip_checks (bool): if True, skip checking completion status and metadata existence.

    Returns:
    proc_dirs (list[str]): list of absolute paths to source data files/archives that need processing.
    """
    from moseq2_extract.helpers.data import check_completion_status

    session_archive_re = re.compile(session_pattern)

    def _filter_unextracted(file_path: Path):
        status_file = None
        metadata_file = None
        is_candidate = False

        # Check for uncompressed session data
        if file_path.suffix == extension and not file_path.name.startswith("ir"):
            status_file = file_path.parent / yaml_path
            metadata_file = file_path.with_name(metadata_path)
            is_candidate = True
        # Check for compressed session archive
        elif session_archive_re.fullmatch(file_path.name):

            session_name = file_path.with_suffix('')
            # Status YAML is expected inside the extracted folder structure
            status_file = session_name / yaml_path
            # Metadata JSON is expected alongside the archive, named after the session
            metadata_file = session_name.with_suffix(".json")
            is_candidate = True

        if not is_candidate: return False # Not a file type we are looking for

        # If skipping checks, any candidate needs processing
        if skip_checks: return True

        # Check if status indicates incomplete and metadata exists
        try:
            is_complete = check_completion_status(status_file)
            return not is_complete and metadata_file.exists()
        except Exception as e:
            warnings.warn(f"Error checking status for {file_path}: {e}. Skipping.")
            return False

    # Use the helper function to find paths needing processing
    proc_dirs = _walk_and_filter(root_dir, _filter_unextracted)

    return proc_dirs

def click_param_annot(click_cmd: click.Command) -> dict[str, str]:
    """
    Return a dict that maps option names to help strings from a click.Command instance.

    Args:
    click_cmd (click.Command): command to annotate

    Returns:
    annotations (dict): dictionary of options and their help messages
    """

    annotations = {}
    for p in click_cmd.params:
        if isinstance(p, click.Option):
            annotations[p.human_readable_name] = p.help
    return annotations

def get_bucket_center(img, true_depth, threshold=650):
    """
    Find Centroid coordinates of circular bucket.

    Args:
    img (np.ndaarray): original background image.
    true_depth (float): distance value from camera to bucket floor (automatically pre-computed)
    threshold (float): distance values to accept region into detected circle. (used to reduce fall noise interference)

    Returns:
    cX (int): x-coordinate of circle centroid
    cY (int): y-coordinate of circle centroid
    """

    # https://stackoverflow.com/questions/19768508/python-opencv-finding-circle-sun-coordinates-of-center-the-circle-from-pictu
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(img, threshold, true_depth, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY
