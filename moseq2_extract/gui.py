"""
GUI front-end operations accessible from a jupyter notebook.
"""

from pathlib import Path
from ast import literal_eval
from moseq2_extract.util import read_yaml, write_yaml
from moseq2_extract.io.image import read_tiff_files
from moseq2_extract.helpers.extract import run_local_extract, run_slurm_extract
from moseq2_extract.helpers.wrappers import (
    get_roi_wrapper,
    extract_wrapper,
    flip_file_wrapper,
    generate_index_wrapper,
    aggregate_extract_results_wrapper,
)
from moseq2_extract.util import (
    recursive_find_unextracted_dirs,
    load_found_session_paths,
    filter_warnings,
)
from moseq2_extract.cli import batch_extract


def get_selected_sessions(to_extract, extract_all):
    """
    Process session selection based on user input or return all sessions.

    Args:
        to_extract (list): List of paths to sessions to extract
        extract_all (bool): If True, return all sessions without prompting user

    Returns:
        list: Paths of selected sessions to extract
    """
    if extract_all or len(to_extract) <= 1:
        return [Path(sess) for sess in to_extract]

    # Display available sessions
    for i, sess in enumerate(to_extract):
        print(f"[{str(i + 1)}] {sess}")

    print("You may input comma separated values for individual sessions")
    print('Or you can input a hyphen separated range. E.g. "1-10" selects 10 sessions, including sessions 1 and 10')
    print('You can also exclude a range by prefixing with "e"; e.g.: "e1-5".')
    print("Press q to quit.")

    selected_sess_idx, excluded_sess_idx = [], []
    ret_extract = []

    def is_valid_int(val):
        try:
            return isinstance(literal_eval(val), int)
        except (ValueError, SyntaxError):
            return False

    def parse_input(s):
        """Parse user input for session selection/exclusion."""
        s = s.strip()
        exclude = False
        
        if s.startswith('e'):
            exclude = True
            s = s[1:].strip()
            
        if '-' in s:
            # Handle range
            start, end = s.split('-', 1)
            if is_valid_int(start) and is_valid_int(end):
                indices = range(int(start), int(end) + 1)
                if exclude:
                    excluded_sess_idx.extend(indices)
                else:
                    selected_sess_idx.extend(indices)
        elif is_valid_int(s):
            # Handle single number
            idx = int(s)
            if exclude:
                excluded_sess_idx.append(idx)
            else:
                selected_sess_idx.append(idx)
    
    while not ret_extract:
        sessions = input("Input your selected sessions to extract: ")
        if "q" in sessions.lower():
            return []
            
        if ',' in sessions:
            # Process comma-separated selections
            for s in sessions.split(','):
                parse_input(s.strip())
        elif sessions:
            # Process single selection
            parse_input(sessions)
            
            # If no sessions selected, assume all
            if not selected_sess_idx:
                selected_sess_idx = list(range(1, len(to_extract) + 1))
        else:
            print("Invalid input. Try again or press q to quit.")
            continue

        # Build final selection list
        for i in selected_sess_idx:
            if i not in excluded_sess_idx and 0 < i <= len(to_extract):
                ret_extract.append(Path(to_extract[i - 1]))
                
    return ret_extract


@filter_warnings
def generate_config_command(output_file, camera_type="k2"):
    """
    Generate configuration file (config.yaml) to use throughout pipeline.

    Args:
    output_file (str): path to saved config file.

    Returns:
    (str): status message.
    """

    from .cli import extract

    objs = extract.params

    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}
    if camera_type == "azure":
        params["bg_roi_depth_range"] = [550, 650]
        params["spatial_filter_size"] = [5]
        params["tail_filter_size"] = [15, 15]

    # Check if the file already exists, and prompt user if they would like to overwrite pre-existing file
    if Path(output_file).exists():
        ow = input(
            "This file already exists, would you like to overwrite it? [y -> yes, n -> no] "
        )
        if ow.lower() == "y":
            # Updating config file
            write_yaml(output_file, params)
        else:
            return "Configuration file has been retained"
    else:
        print("Creating configuration file.")
        write_yaml(output_file, params)

    return "Configuration file has been successfully generated."


@filter_warnings
def extract_found_sessions(
    input_dir, config_file, ext, extract_all=True, skip_extracted=False
):
    """
    Find and extract all depth files with specified extensions within input_dir

    Args:
    input_dir (str): path to directory containing all session folders
    config_file (str): path to config file
    ext (str): file extension for depth files to search for
    extract_all (bool): if True, auto searches for all sessions, else, prompts user to select sessions individually.
    skip_extracted (bool): indicates whether to skip already extracted session.

    """
    # error out early
    if not Path(config_file).exists():
        raise IOError(f"Config file {config_file} does not exist")

    to_extract = []

    # find directories with .dat files that either have incomplete or no extractions
    if isinstance(ext, str):
        ext = [ext]
    for ex in ext:
        tmp = recursive_find_unextracted_dirs(input_dir, extension=ex, skip_checks=True)
        to_extract += [e for e in tmp if e.endswith(ex)]

    # filter out any incorrectly returned sessions
    temp = sorted([sess_dir for sess_dir in to_extract if "/tmp/" not in sess_dir])
    to_extract = get_selected_sessions(temp, extract_all)

    # read in the config file
    config_data = read_yaml(config_file)

    if config_data["cluster_type"] == "local":
        run_local_extract(to_extract, config_file, skip_extracted)
        print("Extractions Complete.")
    else:
        # Get default CLI params
        params = {
            tmp.name: tmp.default for tmp in batch_extract.params if not tmp.required
        }
        # merge default params and config data, preferring values in config data
        config_data = {**params, **config_data}

        # function call to run_slurm_extract to be implemented
        run_slurm_extract(input_dir, to_extract, config_data, skip_extracted)


def generate_index_command(input_dir, output_file):
    """
    Generate Index File (moseq2-index.yaml) based on aggregated sessions

    Args:
    input_dir (str): path to folder with aggregated results
    output_file (str): path to index file

    Returns:
    output_file (str): path to index file.
    """

    output_file = generate_index_wrapper(input_dir, output_file)
    print("Index file successfully generated.")
    return output_file


@filter_warnings
def aggregate_extract_results_command(
    input_dir, format, output_dir, mouse_threshold=0.0
):
    """
    Find all extracted h5, yaml and mp4 files and copies them all to a
    new directory relabeled with their respective session names.
    Also generates the index file (moseq2-index.yaml).

    Args:
    input_dir (str): path to base directory to recursively search for extracted files
    format (str): filename format for info to include in filenames
    output_dir (str): path to directory to save all aggregated results
    mouse_threshold (float): min threshold of extracted mouse height to include a session
    in the aggregated sesssions to ensure sessions with no extracted mouse is not included in the aggregated sessions.


    Returns:
    indexpath (str): path to generated index file (moseq2-index.yaml).
    """

    output_dir = Path(input_dir) / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    indexpath = aggregate_extract_results_wrapper(
        input_dir, format, output_dir, mouse_threshold
    )

    return indexpath


def download_flip_command(output_dir, config_file="", selection=1):
    """
    Download flip classifier and saves its path to config file (config.yaml)

    Args:
    output_dir (str): path to output directory to save flip classifier
    config_file (str): path to config file (config.yaml)
    selection (int): index of which flip file to download (default is Adult male C57 classifer)

    Returns:
    """

    flip_file_wrapper(config_file, output_dir, selected_flip=selection)


@filter_warnings
def find_roi_command(
    input_dir,
    config_file,
    exts=["dat", "avi"],
    select_session=False,
    default_session=0,
):
    """
    Compute ROI files given depth file.

    Args:
    input_dir (str): path to directory containing depth file
    config_file (str): path to config file
    exts (list): list of supported extensions
    select_session (bool): list all found sessions and allow user to select specific session to analyze via user-prompt
    default_session (int): index of the default session to find ROI for

    Returns:
    images (list of 2d arrays): list of 2d array images to graph in Notebook.
    filenames (list): list of paths to respective image paths
    """

    files = load_found_session_paths(input_dir, exts)

    if len(files) == 0:
        print("No recordings found")
        return

    if select_session:
        input_file = get_selected_sessions(files, False)
        if isinstance(input_file, list):
            input_file = input_file[0]
    else:
        input_file = files[default_session]

    print(f"Processing session: {input_file}")
    config_data = read_yaml(config_file)

    output_dir = Path(input_file).parent / "proc"
    get_roi_wrapper(input_file, config_data, output_dir)

    write_yaml(config_file, config_data)

    images, filenames = read_tiff_files(output_dir)

    print(f"ROIs were successfully computed in {output_dir}")
    return images, filenames


@filter_warnings
def extract_command(input_file, output_dir, config_file, num_frames=None, skip=False):
    """
    Extract depth file

    Args:
    input_file (str): path to depth file to extract.
    output_dir (str): path to output directory.
    config_file (str): path to config file (config.yaml).
    num_frames (int): number of frames to extract. If None, all frames are extracted.
    skip (bool): skip already extracted file.

    Returns:
    (str): String indicating that the extracted is completed.
    """
    input_file = Path(input_file)

    config_data = read_yaml(config_file)

    # Loading individual session config parameters if it exists
    if Path(config_data.get("session_config_path", "")).exists():
        session_configs = read_yaml(config_data["session_config_path"])
        session_key = input_file.parent.name

        # If key is found, update config_data, otherwise, use default dict
        config_data = session_configs.get(session_key, config_data)

    if output_dir is None:
        output_dir = config_data.get("output_dir", "proc")

    extract_wrapper(
        input_file, output_dir, config_data, num_frames=num_frames, skip=skip
    )

    return "Extraction completed."
