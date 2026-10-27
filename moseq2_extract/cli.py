"""
CLI for extracting the depth data.
"""

import click
from pathlib import Path
from moseq2_extract.cli_spec import (
    ROI_OPTIONS,
    AVI_OPTIONS,
    EXTRACT_OPTIONS,
    SLURM_OPTIONS,
    option_spec,
)
from moseq2_extract.util import (
    recursive_find_unextracted_dirs,
    read_yaml,
    write_yaml,
    command_with_config,
)
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
from moseq2_extract.flip.train import train_classifier, save_classifier



common_roi_options = option_spec(ROI_OPTIONS)
common_avi_options = option_spec(AVI_OPTIONS)
common_extract_options = option_spec(EXTRACT_OPTIONS)
slurm_options = option_spec(SLURM_OPTIONS)


@click.group(context_settings=dict(show_default=True))
@click.version_option()
def cli():
    """MoSeq2 Extract: Extract mouse behavior from depth videos."""
    pass


@cli.command(
    name="find-roi",
    cls=command_with_config("config_file"),
    help="Finds the ROI (the arena) and background to subtract from frames when extracting.",
)
@click.argument("input-file", type=click.Path(exists=True))
@common_roi_options
def find_roi(input_file, output_dir, **kwargs):
    get_roi_wrapper(input_file, kwargs, output_dir)


@cli.command(
    name="extract",
    cls=command_with_config("config_file"),
    help="Processes raw input depth recordings to output a cropped and oriented "
    "video of the mouse and saves the output+metadata to h5 files in the given output directory.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@common_roi_options
@common_avi_options
@common_extract_options
def extract(input_file, output_dir, num_frames, skip_completed, **kwargs):
    extract_wrapper(
        input_file, output_dir, kwargs, num_frames=num_frames, skip=skip_completed
    )


@cli.command(
    name="batch-extract",
    cls=command_with_config("config_file"),
    help="Batch processes all the raw depth recordings located in the input folder.",
)
@click.argument("input-folder", type=click.Path(exists=True, resolve_path=False))
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Batch command string to prefix model training command (slurm only).",
)
@click.option(
    "--get-cmd", is_flag=True, default=True, help="Print scan command strings."
)
@click.option("--run-cmd", is_flag=True, default=False, help="Run scan command strings.")
@click.option(
    "--extract-out-script",
    type=click.Path(),
    default="extract_out.sh",
    help="Name of bash script file to save extract commands.",
)
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
    default=False,
    help="Flag: skip checks for the existence of a metadata file",
)
@common_roi_options
@common_avi_options
@common_extract_options
@slurm_options
def batch_extract(
    input_folder,
    **kwargs,
):

    output_dir = kwargs["output_dir"]
    skip_completed = kwargs["skip_completed"]

    # check if there is a config file
    config_file = kwargs.get("config_file")
    if not config_file:
        # Add message to tell the users to specify a config file
        print(
            "Command not run. Please specified a config file using --config-file flag."
        )
        return

    to_extract = []
    for ex in kwargs["extensions"]:
        yaml_path = Path(output_dir) / "results_00.yaml"
        to_extract.extend(
            recursive_find_unextracted_dirs(
                input_folder,
                extension=ex,
                skip_checks=kwargs["skip_checks"],
                yaml_path=yaml_path,
            )
        )

    if len(to_extract) == 0:
        print(
            "No new sessions to be extracted. If you want to re-extract, "
            "ensure --skip-completed is set to false or (re)move existing results."
        )
        return

    if kwargs["cluster_type"] == "local":
        # the session specific config doesn't get generated in session proc file
        # session specific config direct used in config_data dictionary in
        # extraction from extract_command function
        run_local_extract(to_extract, config_file, skip_completed)
    else:
        # add parameters to config
        kwargs["session_config_path"] = (
            read_yaml(config_file).get("session_config_path", "")
            if config_file is not None
            else ""
        )
        kwargs["config_file"] = str(Path(config_file).absolute())
        run_slurm_extract(input_folder, to_extract, kwargs, skip_completed)


@cli.command(
    name="download-flip-file",
    help="Downloads Flip-correction model that helps with orienting the mouse during extraction.",
)
@click.argument(
    "config-file",
    type=click.Path(exists=True, resolve_path=False),
    default="config.yaml",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=Path.cwd(),
    help="Output directory for downloaded flip file",
)
def download_flip_file(config_file, output_dir):
    flip_file_wrapper(config_file, output_dir)


@cli.command(
    name="generate-config",
    help="Generates a configuration file (config.yaml) that holds editable options for extraction parameters.",
)
@click.option("--output-file", "-o", type=click.Path(), default="config.yaml")
@click.option(
    "--camera-type",
    default="k2",
    type=str,
    help="specify the camera type (k2 or azure), default is k2",
)
def generate_config(output_file, camera_type):
    params = {
        tmp.name: tmp.default
        for tmp in extract.params
        if isinstance(tmp, click.core.Option) and not tmp.required
    }
    if camera_type == "azure":
        params["bg_roi_depth_range"] = [550, 650]
        params["spatial_filter_size"] = [5]
        params["tail_filter_size"] = [15, 15]
        params["crop_size"] = [120, 120]
        params["camera_type"] = "azure"

    write_yaml(output_file, params)

    print(f"Successfully generated config file in base directory.")


@cli.command(
    name="generate-index",
    help="Generates an index file (moseq2-index.yaml) that contains all extracted session metadata.",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=Path.cwd(),
    help="Directory to find h5 files",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    default=Path.cwd() / "moseq2-index.yaml",
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
    default=Path.cwd(),
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
    default=Path.cwd() / "aggregate_results",
    help="Directory for storing aggregated extraction results",
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
    default=Path.cwd() / "aggregate_results",
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
    input_file,
    output_file,
    chunk_size,
    fps,
    delete,
):

    convert_raw_to_avi_wrapper(
        input_file, output_file, chunk_size, fps, delete
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
):

    copy_slice_wrapper(
        input_file, output_file, copy_slice, chunk_size, fps, delete
    )

@cli.command(
    name="train-flip-classifier",
    help="Train a classifier to predict the orientation of a mouse.",
)
@click.option("--data-path", type=click.Path(exists=True, resolve_path=False), required=True, help="Path to the training data numpy file.")
@click.option("--classifier", type=click.Choice(["SVM", "RF"]), default="SVM", help="Classifier to use.")
@click.option("--n-components", type=int, default=20, help="Number of components to keep in PCA.")
def train_flip_classifier(data_path, classifier, n_components):
    clf = train_classifier(data_path, classifier, n_components)
    save_classifier(clf, f"flip_classifier_{classifier}.p")



if __name__ == "__main__":
    cli()
