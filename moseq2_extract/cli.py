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
from moseq2_extract.util import recursive_find_unextracted_dirs, read_yaml
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


def load_config(ctx, param, value):
    """Callback to load configuration from a yaml file and set defaults."""
    if not value or not Path(value).exists():
        return None  # No config file specified or found

    try:
        config = read_yaml(value)
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
    help="Path to a yaml configuration file. Options defined here are overridden by CLI arguments.",
    callback=load_config,
    is_eager=True,
)
def cli(config_file):
    """MoSeq2 Extract: Extract mouse behavior from depth videos."""
    pass


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
@common_extract_options
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
    "--prefix",
    type=str,
    default="",
    help="Batch command string to prefix model training command (slurm only).",
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
@common_roi_options
@common_avi_options
@common_extract_options
@slurm_options
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
    default=Path.cwd(),
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
@click.option("--output-file", "-o", type=click.Path(), default="config.yaml")
@click.option(
    "--camera-type",
    default="k2",
    type=click.Choice(["k2", "azure", "auto"]),
    help="specify the camera type (k2 or azure), default is k2",
)
def generate_config(output_file, camera_type):
    """Copy default config and patch selected fields via sed to keep comments/structure."""
    import shutil

    script_path = Path(__file__).parent
    default_path = script_path / "default-config.yaml"
    shutil.copy(default_path, output_file)

    if camera_type == "azure":
        replacements = [
            ("bg_roi_depth_range", "[550, 650]"),
            ("spatial_filter_size", "[5]"),
            ("tail_filter_size", "[15, 15]"),
            ("crop_size", "[120, 120]"),
            ("camera_type", '"azure"'),
        ]

        with open(output_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            for key, val in replacements:
                if key in line:
                    line = f"  {key}: {val}\n"
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
