"""
Image reading/writing functionality.
"""

import ast
import json
import numpy as np
import imageio.v3 as iio
from pathlib import Path


def read_tiff_files(input_dir):
    """
    Read ROI output results (Tiff files) located in the given input_directory.

    Args:
    input_dir (str): path to directory containing ROI files.

    Returns:
    images (list): list of 2d arrays of the ROIs.
    filenames (list): list of corresponding filenames to each read image.
    """

    images = []
    filenames = []
    input_path = Path(input_dir)
    for infile in input_path.iterdir():
        if infile.name.endswith("tiff"):
            im = read_tiff(input_path / infile.name)
            if len(im.shape) == 2:
                images.append(im)
            elif len(im.shape) == 3:
                images.append(im[0])
            filenames.append(infile.name)

    return images, filenames


def write_tiff(
    filename, image, scale=True, scale_factor=None, frame_dtype="uint16", compress=0
):
    """
    Save image data.

    Args:
    filename (str): path to output file
    image (numpy.ndarray): the (unscaled) 2-D image to save
    scale (bool): flag to scale the image between the bounds of `dtype`
    scale_factor (int): factor by which to scale image
    frame_dtype (str): array data type
    compress (int): image compression level
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    metadata = None

    if scale:
        max_int = np.iinfo(frame_dtype).max

        if not scale_factor:
            # scale image to `dtype`'s full range
            scale_factor = int(
                max_int / (np.nanmax(image) + 1e-25)
            )  # adding very small value to avoid divide by 0
            image = image * scale_factor
        elif isinstance(scale_factor, tuple):
            image = np.float32(image)
            image = (image - scale_factor[0]) / (scale_factor[1] - scale_factor[0])
            image = np.clip(image, 0, 1) * max_int

        metadata = json.dumps({"scale_factor": str(scale_factor)})

    # embed scale_factor metadata in the TIFF description and write with imageio
    iio.imwrite(filename, image.astype(frame_dtype), compression=compress, description=metadata)


def read_tiff(filename, scale=True, scale_key="scale_factor"):
    """
    Load image data

    Args:
    filename (str): path to output file
    scale (bool): flag that indicates whether to scale image
    scale_key (str): indicates scale factor.

    Returns:
    image (numpy.ndarray): loaded image
    """

    # use imageio to read TIFF and extract metadata
    with iio.imopen(filename, "r") as reader:
        desc = reader.metadata(index=0).get('description')
        image = reader.read()

    if scale:
        # parse embedded description JSON metadata
        image_desc = json.loads(desc) if desc else {}

        try:
            scale_factor = int(image_desc[scale_key])
            image = image / scale_factor
        except ValueError:
            scale_factor = ast.literal_eval(image_desc[scale_key])

        if isinstance(scale_factor, tuple):
            iinfo = np.iinfo(image.dtype)
            image = image.astype("float32") / iinfo.max
            image = image * (scale_factor[1] - scale_factor[0]) + scale_factor[0]

    return image
