import click


ROI_OPTIONS = [
    (
        ["--bg-roi-dilate"],
        {
            "default": (10, 10),
            "type": (int, int),
            "help": "Size of StructuringElement to dilate roi",
        },
    ),
    (
        ["--bg-roi-shape"],
        {
            "default": "ellipse",
            "type": str,
            "help": "Shape to use to detect roi (ellipse or rect)",
        },
    ),
    (
        ["--bg-roi-index"],
        {"default": 0, "type": int, "help": "Index of which detected ROI mask to use"},
    ),
    (
        ["--bg-roi-weights"],
        {
            "default": (1, 0.1, 1),
            "type": (float, float, float),
            "help": "ROI feature weighting (area, extent, dist to center)",
        },
    ),
    (
        ["--camera-type"],
        {
            "default": "auto",
            "type": click.Choice(["auto", "kinect", "azure", "realsense"]),
            "help": "Camera type used for recording for auto-sets bg-roi-weights to precomputed values for different camera types.",
        },
    ),
    (
        ["--manual-set-depth-range"],
        {"is_flag": True, "help": "Flag to deactivate auto depth range setting."},
    ),
    (
        ["--bg-roi-depth-range"],
        {
            "default": (650, 750),
            "type": (float, float),
            "help": "Range to search for floor of arena (in mm)",
        },
    ),
    (
        ["--bg-roi-gradient-filter"],
        {
            "default": False,
            "type": bool,
            "help": "Use gradient filter to exclude walls for detected ROI",
        },
    ),
    (
        ["--bg-roi-gradient-threshold"],
        {
            "default": 3000,
            "type": float,
            "help": "Gradient must be less than threshold to include points",
        },
    ),
    (
        ["--bg-roi-gradient-kernel"],
        {"default": 7, "type": int, "help": "Kernel size for Sobel gradient filtering"},
    ),
    (
        ["--bg-roi-fill-holes"],
        {"default": True, "type": bool, "help": "Fill holes in ROI"},
    ),
    (
        ["--bg-sort-roi-by-position"],
        {"default": False, "type": bool, "help": "Sort ROIs by position"},
    ),
    (
        ["--bg-sort-roi-by-position-max-rois"],
        {
            "default": 2,
            "type": int,
            "help": "The number of maximum ROIs sorted by area",
        },
    ),
    (
        ["--dilate-iterations"],
        {
            "default": 1,
            "type": int,
            "help": "Number of dilation iterations to increase bucket floor size.",
        },
    ),
    (
        ["--bg-roi-erode"],
        {
            "default": (1, 1),
            "type": (int, int),
            "help": "Size of cv2 Structure Element to erode roi.",
        },
    ),
    (
        ["--bg-v2"],
        {
            "is_flag": True,
            "help": "Flag to adaptively use best quantile for computing background",
        },
    ),
    (
        ["--erode-iterations"],
        {
            "default": 0,
            "type": int,
            "help": "Number of erosion iterations to decrease bucket floor size.",
        },
    ),
    (
        ["--noise-tolerance"],
        {
            "default": 30,
            "type": int,
            "help": "Extent of noise to accept during RANSAC Plane ROI computation.",
        },
    ),
    (
        ["--output-dir"],
        {"default": "proc", "help": "Output directory to save the results h5 file"},
    ),
    (
        ["--use-plane-bground"],
        {
            "is_flag": True,
            "help": "Use a plane fit for the background. Useful when mice don't move much",
        },
    ),
    (
        ["--recompute-bg"],
        {"default": False, "help": "Overwrite previously computed background image"},
    ),
    (
        ["--progress-bar", "-p"],
        {"is_flag": True, "help": "Show verbose progress bars."},
    ),
]

AVI_OPTIONS = [
    (
        ["-o", "--output-file"],
        {"type": click.Path(), "default": None, "help": "Path to output file"},
    ),
    (["-b", "--chunk-size"], {"type": int, "default": 3000, "help": "Chunk size"}),
    (["--fps"], {"type": float, "default": 30, "help": "Video FPS"}),
    (
        ["--delete"],
        {"is_flag": True, "help": "Delete raw file if encoding is successful"},
    ),
    (
        ["-t", "--threads"],
        {
            "type": int,
            "default": 8,
            "help": "Number of threads used saving ffv1 endcoded AVI file with ffmpeg",
        },
    ),
    (
        ["-m", "--mapping"],
        {
            "type": str,
            "default": "DEPTH",
            "help": "Ffprobe stream selection variable. Default: DEPTH",
        },
    ),
]

EXTRACT_OPTIONS = [
    (
        ["--crop-size", "-c"],
        {
            "default": (80, 80),
            "type": (int, int),
            "help": "Width and height of cropped mouse image",
        },
    ),
    (
        ["--num-frames", "-n"],
        {
            "default": None,
            "type": int,
            "help": "Number of frames to extract. Will extract full session if set to None.",
        },
    ),
    (
        ["--min-height"],
        {
            "default": 10,
            "type": int,
            "help": "Min mouse height threshold from floor (mm)",
        },
    ),
    (
        ["--max-height"],
        {
            "default": 120,
            "type": int,
            "help": "Max mouse height threshold from floor (mm)",
        },
    ),
    (
        ["--detected-true-depth"],
        {
            "default": "auto",
            "type": str,
            "help": "Option to override automatic depth estimation during extraction.",
        },
    ),
    (
        ["--compute-raw-scalars"],
        {"is_flag": True, "help": "Compute scalar values from raw cropped frames."},
    ),
    (
        ["--flip-classifier"],
        {
            "default": None,
            "help": "Path to the flip classifier used to properly orient the mouse (.pkl file)",
        },
    ),
    (
        ["--flip-classifier-smoothing"],
        {
            "default": 51,
            "type": int,
            "help": "Number of frames to smooth flip classifier probabilities",
        },
    ),
    (
        ["--graduate-walls"],
        {
            "default": False,
            "type": bool,
            "help": "Graduates and dilates the background image to compensate for slanted bucket walls.",
        },
    ),
    (
        ["--widen-radius"],
        {
            "default": 0,
            "type": int,
            "help": "Number of pixels to increase/decrease radius by when graduating bucket walls.",
        },
    ),
    (
        ["--use-cc"],
        {
            "default": True,
            "type": bool,
            "help": "Extract features using largest connected components.",
        },
    ),
    (
        ["--use-tracking-model"],
        {
            "default": False,
            "type": bool,
            "help": "Use an expectation-maximization style model to aid mouse tracking. Useful for data with cables",
        },
    ),
    (
        ["--tracking-model-ll-threshold"],
        {
            "default": -100,
            "type": float,
            "help": "Threshold on log-likelihood for pixels to use for update during tracking",
        },
    ),
    (
        ["--tracking-model-mask-threshold"],
        {
            "default": -16,
            "type": float,
            "help": "Threshold on log-likelihood to include pixels for centroid and angle calculation",
        },
    ),
    (
        ["--tracking-model-ll-clip"],
        {
            "default": -100,
            "type": float,
            "help": "Clip log-likelihoods below this value",
        },
    ),
    (
        ["--tracking-model-segment"],
        {
            "default": True,
            "type": bool,
            "help": "Segment likelihood mask from tracking model",
        },
    ),
    (
        ["--tracking-model-init"],
        {
            "default": "raw",
            "type": str,
            "help": "Method for tracking model initialization",
        },
    ),
    (
        ["--cable-filter-iters"],
        {"default": 0, "type": int, "help": "Number of cable filter iterations"},
    ),
    (
        ["--cable-filter-shape"],
        {
            "default": "rectangle",
            "type": click.Choice(["rectangle", "ellipse"]),
            "help": "Cable filter shape (rectangle or ellipse)",
        },
    ),
    (
        ["--cable-filter-size"],
        {
            "default": (5, 5),
            "type": (int, int),
            "help": "Cable filter size (in pixels)",
        },
    ),
    (
        ["--tail-filter-iters"],
        {"default": 1, "type": int, "help": "Number of tail filter iterations"},
    ),
    (
        ["--tail-filter-size"],
        {"default": (9, 9), "type": (int, int), "help": "Tail filter size"},
    ),
    (
        ["--tail-filter-shape"],
        {
            "default": "ellipse",
            "type": click.Choice(["rectangle", "ellipse"]),
            "help": "Tail filter shape",
        },
    ),
    (
        ["--spatial-filter-size", "-s"],
        {
            "default": [3],
            "type": int,
            "help": "Space prefilter kernel (median filter, must be odd)",
            "multiple": True,
        },
    ),
    (
        ["--temporal-filter-size"],
        {
            "default": [0],
            "type": int,
            "help": "Time prefilter kernel (median filter, must be odd)",
            "multiple": True,
        },
    ),
    (
        ["--chunk-overlap"],
        {
            "default": 0,
            "type": int,
            "help": "Frames overlapped in each chunk. Useful for cable tracking",
        },
    ),
    (
        ["--write-movie"],
        {
            "default": True,
            "type": bool,
            "help": "Write a results output movie including an extracted mouse",
        },
    ),
    (
        ["--frame-dtype"],
        {
            "default": "uint8",
            "type": click.Choice(["uint8", "uint16"]),
            "help": "Data type for processed frames",
        },
    ),
    (
        ["--movie-dtype"],
        {"default": "<i2", "help": "Data type for raw frames read in for extraction"},
    ),
    (
        ["--pixel-format"],
        {
            "default": "gray16le",
            "type": str,
            "help": "Pixel format for reading in .avi and .mkv videos",
        },
    ),
    (
        ["--centroid-hampel-span"],
        {"default": 0, "type": int, "help": "Hampel filter span"},
    ),
    (
        ["--centroid-hampel-sig"],
        {"default": 3, "type": float, "help": "Hampel filter sig"},
    ),
    (["--angle-hampel-span"], {"default": 0, "type": int, "help": "Angle filter span"}),
    (["--angle-hampel-sig"], {"default": 3, "type": float, "help": "Angle filter sig"}),
    (
        ["--model-smoothing-clips"],
        {"default": (0, 0), "type": (float, float), "help": "Model smoothing clips"},
    ),
    (
        ["--frame-trim"],
        {
            "default": (0, 0),
            "type": (int, int),
            "help": "Frames to trim from beginning and end of data",
        },
    ),
    (
        ["--compress"],
        {
            "default": False,
            "type": bool,
            "help": "Convert .dat to .avi after successful extraction",
        },
    ),
    (
        ["--compress-chunk-size"],
        {"type": int, "default": 3000, "help": "Chunk size for .avi compression"},
    ),
    (
        ["--compress-threads"],
        {"type": int, "default": 3, "help": "Number of threads for encoding"},
    ),
    (
        ["--skip-completed"],
        {
            "is_flag": True,
            "help": "Will skip the extraction if it is already completed.",
        },
    ),
]

SLURM_OPTIONS = [
    (["--cluster-type"], {
        "type": click.Choice(["local","slurm"]), "default":"local",
        "help":"Platform to train models on"
    }),
    (["-c","--ncpus"],  {"type":int, "default":1, "help":"# cores"}),
    (["--memory"],      {"type":str, "default":"5GB", "help":"RAM (slurm)"}),
    (["--wall-time"],   {"type":str, "default":"3:00:00","help":"Wall time"}),
    (["--partition"],   {"type":str, "default":"short","help":"Slurm partition"}),
]


def option_spec(spec_list: list[tuple]):
    """Build a decorator that applies all click.option entries from the given list."""

    def decorator(fn):
        for args, kwargs in reversed(spec_list):
            fn = click.option(*args, **kwargs)(fn)
        return fn

    return decorator
