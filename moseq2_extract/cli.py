"""
CLI for extracting the depth data.
"""

import os
import toml
import click
from moseq2_extract.util import recursive_find_unextracted_dirs
from moseq2_extract.helpers.wrappers import (
    get_roi_wrapper,
    extract_wrapper,
    flip_file_wrapper,
    generate_index_wrapper,
    aggregate_extract_results_wrapper,
    generate_index_from_agg_res_wrapper,
    convert_raw_to_avi_wrapper,
    copy_slice_wrapper,
)
from moseq2_extract.helpers.extract import run_slurm_extract, run_local_extract
from pathlib import Path


def load_config(ctx, param, value):
    """Callback to load configuration from a TOML file and set defaults."""
    if not value or not os.path.exists(value):
        return None  # No config file specified or found

    try:
        with open(value, "r") as f:
            config = toml.load(f)
        # Extract the [extract] section if it exists
        extract_config = config.get("extract", {})
        if not isinstance(extract_config, dict):
            raise click.BadParameter("Config [extract] section must be a dictionary.")

        # Set the default map for the context if it doesn't exist
        ctx.default_map = ctx.default_map or {}
        # Update the default map with values from the config file
        # add default map to each subcommand
        _maps = {}
        for command in ctx.command.commands.values():
            _maps[command.name] = extract_config
        ctx.default_map.update(_maps)

        ctx.ensure_object(dict).update({"config_path": value})

    except Exception as e:
        raise click.BadParameter(f"Error parsing config file {value}: {e}")

    return value  # Return the path itself


@click.group(context_settings=dict(show_default=True, default_map={}))
@click.version_option()
@click.option(
    "--config-file",
    type=click.Path(dir_okay=False),
    help="Path to a TOML configuration file. Options defined here are overridden by CLI arguments.",
    callback=load_config,
    is_eager=True,
)
def cli(config_file):
    """MoSeq2 Extract: Extract mouse behavior from depth videos."""
    pass


def common_roi_options(function):
    """
    Decorator function for grouping shared ROI related parameters.
    Defaults are set to None, allowing config file or ultimate defaults to apply.
    """
    function = click.option(
        "--bg-roi-dilate",
        default=(10, 10),
        type=(int, int),
        help="Size of StructuringElement to dilate roi",
    )(function)
    function = click.option(
        "--bg-roi-shape",
        default="ellipse",
        type=str,
        help="Shape to use to detect roi (ellipse or rect)",
    )(function)
    function = click.option(
        "--bg-roi-index",
        default=0,
        type=int,
        help="Index of which detected ROI mask to use",
    )(function)
    function = click.option(
        "--bg-roi-weights",
        default=(1, 0.1, 1),
        type=(float, float, float),
        help="ROI feature weighting (area, extent, dist to center)",
    )(function)
    function = click.option(
        "--camera-type",
        default="auto",
        type=click.Choice(["auto", "kinect", "azure", "realsense"]),
        help="Camera type used for recording for auto-sets bg-roi-weights to precomputed values for different camera types.",
    )(function)
    function = click.option(
        "--manual-set-depth-range",
        is_flag=True,
        help="Flag to deactivate auto depth range setting.",
    )(function)
    function = click.option(
        "--bg-roi-depth-range",
        default=(650, 750),
        type=(float, float),
        help="Range to search for floor of arena (in mm)",
    )(function)
    function = click.option(
        "--bg-roi-gradient-filter",
        default=False,
        type=bool,
        help="Use gradient filter to exclude walls for detected ROI",
    )(function)
    function = click.option(
        "--bg-roi-gradient-threshold",
        default=3000,
        type=float,
        help="Gradient must be less than threshold to include points",
    )(function)
    function = click.option(
        "--bg-roi-gradient-kernel",
        default=7,
        type=int,
        help="Kernel size for Sobel gradient filtering",
    )(function)
    function = click.option(
        "--bg-roi-fill-holes", default=True, type=bool, help="Fill holes in ROI"
    )(function)
    function = click.option(
        "--bg-sort-roi-by-position",
        default=False,
        type=bool,
        help="Sort ROIs by position",
    )(function)
    function = click.option(
        "--bg-sort-roi-by-position-max-rois",
        default=2,
        type=int,
        help="The number of maximum ROIs sorted by area",
    )(function)
    function = click.option(
        "--dilate-iterations",
        default=1,
        type=int,
        help="Number of dilation iterations to increase bucket floor size.",
    )(function)
    function = click.option(
        "--bg-roi-erode",
        default=(1, 1),
        type=(int, int),
        help="Size of cv2 Structure Element to erode roi.",
    )(function)
    function = click.option(
        "--bg-v2",
        is_flag=True,
        help="Flag to adaptively use best quantile for computing background",
    )(function)
    function = click.option(
        "--erode-iterations",
        default=0,
        type=int,
        help="Number of erosion iterations to decrease bucket floor size.",
    )(function)
    function = click.option(
        "--noise-tolerance",
        default=30,
        type=int,
        help="Extent of noise to accept during RANSAC Plane ROI computation.",
    )(function)
    function = click.option(
        "--output-dir",
        default="proc",
        help="Output directory to save the results h5 file",
    )(function)
    function = click.option(
        "--use-plane-bground",
        is_flag=True,
        help="Use a plane fit for the background. Useful when mice don't move much",
    )(function)
    function = click.option(
        "--recompute-bg",
        default=False,
        help="Overwrite previously computed background image",
    )(function)
    function = click.option(
        "--progress-bar", "-p", is_flag=True, help="Show verbose progress bars."
    )(function)
    return function


def common_avi_options(function):
    """
    Decorator function for grouping shared video processing parameters.
    Defaults are set to None, allowing config file or ultimate defaults to apply.
    """
    function = click.option(
        "-o",
        "--output-file",
        type=click.Path(),
        default=None,
        help="Path to output file",
    )(function)
    function = click.option(
        "-b", "--chunk-size", type=int, default=3000, help="Chunk size"
    )(function)
    function = click.option("--fps", type=float, default=30, help="Video FPS")(function)
    function = click.option(
        "--delete", is_flag=True, help="Delete raw file if encoding is successful"
    )(function)
    function = click.option(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Number of threads used saving ffv1 endcoded AVI file with ffmpeg",
    )(function)
    function = click.option(
        "-m",
        "--mapping",
        type=str,
        default="DEPTH",
        help="Ffprobe stream selection variable. Default: DEPTH",
    )(function)

    return function


def extract_options(function):
    """
    Decorator function for grouping shared extraction parameters.
    Defaults are set to None, allowing config file or ultimate defaults to apply.
    """
    function = click.option(
        "--crop-size",
        "-c",
        default=(80, 80),
        type=(int, int),
        help="Width and height of cropped mouse image",
    )(function)
    function = click.option(
        "--num-frames",
        "-n",
        default=None,
        type=int,
        help="Number of frames to extract. Will extract full session if set to None.",
    )(function)
    function = click.option(
        "--min-height",
        default=10,
        type=int,
        help="Min mouse height threshold from floor (mm)",
    )(function)
    function = click.option(
        "--max-height",
        default=120,
        type=int,
        help="Max mouse height threshold from floor (mm)",
    )(function)
    function = click.option(
        "--detected-true-depth",
        default="auto",
        type=str,
        help="Option to override automatic depth estimation during extraction.",
    )(function)
    function = click.option(
        "--compute-raw-scalars",
        is_flag=True,
        help="Compute scalar values from raw cropped frames.",
    )(function)
    function = click.option(
        "--flip-classifier",
        default=None,
        help="Path to the flip classifier used to properly orient the mouse (.pkl file)",
    )(function)
    function = click.option(
        "--flip-classifier-smoothing",
        default=51,
        type=int,
        help="Number of frames to smooth flip classifier probabilities",
    )(function)
    function = click.option(
        "--graduate-walls",
        default=False,
        type=bool,
        help="Graduates and dilates the background image to compensate for slanted bucket walls.",
    )(function)
    function = click.option(
        "--widen-radius",
        default=0,
        type=int,
        help="Number of pixels to increase/decrease radius by when graduating bucket walls.",
    )(function)
    function = click.option(
        "--use-cc",
        default=True,
        type=bool,
        help="Extract features using largest connected components.",
    )(function)
    function = click.option(
        "--use-tracking-model",
        default=False,
        type=bool,
        help="Use an expectation-maximization style model to aid mouse tracking. Useful for data with cables",
    )(function)
    function = click.option(
        "--tracking-model-ll-threshold",
        default=-100,
        type=float,
        help="Threshold on log-likelihood for pixels to use for update during tracking",
    )(function)
    function = click.option(
        "--tracking-model-mask-threshold",
        default=-16,
        type=float,
        help="Threshold on log-likelihood to include pixels for centroid and angle calculation",
    )(function)
    function = click.option(
        "--tracking-model-ll-clip",
        default=-100,
        type=float,
        help="Clip log-likelihoods below this value",
    )(function)
    function = click.option(
        "--tracking-model-segment",
        default=True,
        type=bool,
        help="Segment likelihood mask from tracking model",
    )(function)
    function = click.option(
        "--tracking-model-init",
        default="raw",
        type=str,
        help="Method for tracking model initialization",
    )(function)
    function = click.option(
        "--cable-filter-iters",
        default=0,
        type=int,
        help="Number of cable filter iterations",
    )(function)
    function = click.option(
        "--cable-filter-shape",
        default="rectangle",
        type=click.Choice(["rectangle", "ellipse"]),
        help="Cable filter shape (rectangle or ellipse)",
    )(function)
    function = click.option(
        "--cable-filter-size",
        default=(5, 5),
        type=(int, int),
        help="Cable filter size (in pixels)",
    )(function)
    function = click.option(
        "--tail-filter-iters",
        default=1,
        type=int,
        help="Number of tail filter iterations",
    )(function)
    function = click.option(
        "--tail-filter-size", default=(9, 9), type=(int, int), help="Tail filter size"
    )(function)
    function = click.option(
        "--tail-filter-shape",
        default="ellipse",
        type=click.Choice(["rectangle", "ellipse"]),
        help="Tail filter shape",
    )(function)
    function = click.option(
        "--spatial-filter-size",
        "-s",
        default=[3],
        type=int,
        help="Space prefilter kernel (median filter, must be odd)",
        multiple=True,
    )(function)
    function = click.option(
        "--temporal-filter-size",
        default=[0],
        type=int,
        help="Time prefilter kernel (median filter, must be odd)",
        multiple=True,
    )(function)
    function = click.option(
        "--chunk-overlap",
        default=0,
        type=int,
        help="Frames overlapped in each chunk. Useful for cable tracking",
    )(function)
    function = click.option(
        "--write-movie",
        default=True,
        type=bool,
        help="Write a results output movie including an extracted mouse",
    )(function)
    function = click.option(
        "--frame-dtype",
        default="uint8",
        type=click.Choice(["uint8", "uint16"]),
        help="Data type for processed frames",
    )(function)
    function = click.option(
        "--movie-dtype",
        default="<i2",
        help="Data type for raw frames read in for extraction",
    )(function)
    function = click.option(
        "--pixel-format",
        default="gray16le",
        type=str,
        help="Pixel format for reading in .avi and .mkv videos",
    )(function)
    function = click.option(
        "--centroid-hampel-span", default=0, type=int, help="Hampel filter span"
    )(function)
    function = click.option(
        "--centroid-hampel-sig", default=3, type=float, help="Hampel filter sig"
    )(function)
    function = click.option(
        "--angle-hampel-span", default=0, type=int, help="Angle filter span"
    )(function)
    function = click.option(
        "--angle-hampel-sig", default=3, type=float, help="Angle filter sig"
    )(function)
    function = click.option(
        "--model-smoothing-clips",
        default=(0, 0),
        type=(float, float),
        help="Model smoothing clips",
    )(function)
    function = click.option(
        "--frame-trim",
        default=(0, 0),
        type=(int, int),
        help="Frames to trim from beginning and end of data",
    )(function)
    function = click.option(
        "--compress",
        default=False,
        type=bool,
        help="Convert .dat to .avi after successful extraction",
    )(function)
    function = click.option(
        "--compress-chunk-size",
        type=int,
        default=3000,
        help="Chunk size for .avi compression",
    )(function)
    function = click.option(
        "--compress-threads", type=int, default=3, help="Number of threads for encoding"
    )(function)
    function = click.option(
        "--skip-completed",
        is_flag=True,
        help="Will skip the extraction if it is already completed.",
    )(function)

    return function


@cli.command(
    name="find-roi",
    help="Finds the ROI (the arena) and background to subtract from frames when extracting.",
)
@click.argument("input-file", type=click.Path(exists=True))
@common_roi_options
def find_roi(input_file, output_dir, **kwargs):
    get_roi_wrapper(input_file, kwargs, output_dir)


@cli.command(
    name="extract",
    help="Processes raw input depth recordings to output a cropped and oriented "
    "video of the mouse and saves the output+metadata to h5 files in the given output directory.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@common_roi_options
@common_avi_options
@extract_options
def extract(input_file, output_dir, num_frames, skip_completed, **kwargs):
    extract_wrapper(
        input_file, output_dir, kwargs, num_frames=num_frames, skip=skip_completed
    )


@cli.command(
    name="batch-extract",
    help="Batch processes all the raw depth recordings located in the input folder.",
)
@click.argument("input-folder", type=click.Path(exists=True, resolve_path=False))
@click.option(
    "--cluster-type",
    type=click.Choice(["local", "slurm"]),
    default="local",
    help="Platform to train models on",
)
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Batch command string to prefix model training command (slurm only).",
)
@click.option(
    "--ncpus", "-c", type=int, default=1, help="Number of cores to use in extraction"
)
@click.option("--memory", type=str, default="5GB", help="RAM (slurm only)")
@click.option("--wall-time", type=str, default="3:00:00", help="Wall time (slurm only)")
@click.option(
    "--partition", type=str, default="short", help="Partition name (slurm only)"
)
@click.option(
    "--get-cmd", is_flag=True, default=True, help="Print scan command strings."
)
@click.option("--run-cmd", is_flag=True, help="Run scan command strings.")
@click.option(
    "--extract-out-script",
    type=click.Path(),
    default="extract_out.sh",
    help="Name of bash script file to save extract commands.",
)
@common_roi_options
@common_avi_options
@extract_options
@click.option(
    "--extensions",
    default=[".dat"],
    type=str,
    help="File extension of raw data",
    multiple=True,
)
@click.option(
    "--skip-checks",
    is_flag=True,
    help="Flag: skip checks for the existence of a metadata file",
)
@click.pass_context
def batch_extract(
    ctx,
    input_folder,
    **kwargs,
):

    output_dir = kwargs["output_dir"]
    skip_completed = kwargs["skip_completed"]

    to_extract = []
    for ex in kwargs["extensions"]:
        yaml_path = os.path.join(output_dir, "results_00.yaml")
        to_extract.extend(
            recursive_find_unextracted_dirs(
                input_folder,
                extension=ex,
                skip_checks=ex in (".tgz", ".tar.gz") or kwargs["skip_checks"],
                yaml_path=yaml_path,
            )
        )

    if len(to_extract) == 0:
        print(
            "No new sessions to be extracted. If you want to re-extract, "
            "ensure --skip-completed is set to false or (re)move existing results."
        )
        return

    config_path = ctx.obj.get("config_path")
    if kwargs["cluster_type"] == "local":
        run_local_extract(to_extract, config_path, skip_completed)
    else:
        # TODO: see if "session_config_path" is defined
        kwargs["config_file"] = config_path
        run_slurm_extract(input_folder, to_extract, kwargs, skip_completed)


@cli.command(
    name="download-flip-file",
    help="Downloads Flip-correction model that helps with orienting the mouse during extraction.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=os.getcwd(),
    help="Output directory for downloaded flip file",
)
@click.pass_context
def download_flip_file(ctx, output_dir):
    # TODO: test fix - get config file path
    config_path = ctx.obj.get("config_path")
    flip_file_wrapper(config_path, output_dir)


@cli.command(
    name="generate-config",
    help="Generates a configuration file (config.yaml) that holds editable options for extraction parameters.",
)
@click.option("--output-file", "-o", type=click.Path(), default="config.toml")
@click.option(
    "--camera-type",
    default="k2",
    type=click.Choice(["k2", "azure", "auto"]),
    help="specify the camera type (k2 or azure), default is k2",
)
def generate_config(output_file, camera_type):
    """Copy default TOML and patch selected fields via sed to keep comments/structure."""
    import shutil

    script_path = Path(__file__).parent
    default_path = script_path / "default-config.toml"
    shutil.copy(default_path, output_file)

    if camera_type == "azure":
        replacements = [
            ("bg_roi_depth_range", "[ 550, 650 ]"),
            ("spatial_filter_size", "[ 5 ]"),
            ("tail_filter_size", "[ 15, 15 ]"),
            ("crop_size", "[ 120, 120 ]"),
            ("camera_type", '"azure"'),
        ]

        with open(output_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            for key, val in replacements:
                if key in line:
                    line = f"{key} = {val}\n"
                    break
            new_lines.append(line)

        with open(output_file, "w") as f:
            f.writelines(new_lines)

    print(f"Successfully generated config file at {output_file}.")


@cli.command(
    name="generate-index",
    help="Generates an index file (moseq2-index.yaml) that contains all extracted session metadata.",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.getcwd(),
    help="Directory to find h5 files",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "moseq2-index.yaml"),
    help="Location for storing index",
)
def generate_index(input_dir, output_file):
    generated_file = generate_index_wrapper(input_dir, output_file)

    if generated_file is not None:
        print(f"Index file: {generated_file} was successfully generated.")


@cli.command(
    name="aggregate-results",
    help="Copies all extracted results (h5, yaml, mp4) files from all extracted sessions to a new directory for modeling and analysis",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.getcwd(),
    help="Directory to find h5 files",
)
@click.option(
    "--format",
    "-f",
    type=str,
    default="{start_time}_{session_name}_{subject_name}",
    help="New file name formats from respective metadata",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "aggregate_results/"),
    help="Location for storing all results together",
)
@click.option(
    "--mouse-threshold",
    default=0,
    type=float,
    help="Threshold value for mean depth to include frames in aggregated results",
)
def aggregate_extract_results(input_dir, format, output_dir, mouse_threshold):

    aggregate_extract_results_wrapper(input_dir, format, output_dir, mouse_threshold)


@cli.command(
    name="agg-to-index",
    help="Generate an index file from aggregated results with default as group names",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "aggregate_results"),
    help="Directory for aggregated results folder",
)
def agg_to_index(input_dir):
    generate_index_from_agg_res_wrapper(input_dir)


@cli.command(
    name="convert-raw-to-avi",
    help="Loss less compresses a raw depth file (dat) into an avi file that is 8x smaller.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@common_avi_options
def convert_raw_to_avi(
    input_file, output_file, chunk_size, fps, delete, threads, mapping,
):

    convert_raw_to_avi_wrapper(
        input_file, output_file, chunk_size, fps, delete, threads, mapping
    )


@cli.command(
    name="copy-slice",
    help="Copies a segment of an input depth recording into a new video file.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@common_avi_options
@click.option(
    "-c",
    "--copy-slice",
    type=(int, int),
    default=(0, 1000),
    help="Slice indices used for copy",
)
def copy_slice(
    input_file,
    output_file,
    copy_slice,
    chunk_size,
    fps,
    delete,
    threads,
    mapping,
):

    copy_slice_wrapper(
        input_file, output_file, copy_slice, chunk_size, fps, delete, threads, mapping
    )


if __name__ == "__main__":
    cli()
