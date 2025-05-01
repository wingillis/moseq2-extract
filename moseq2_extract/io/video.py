"""
Video and video-metadata read/write functions.
"""

import av
import os
import cv2
import subprocess
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from cytoolz import partition_all
from collections import deque
from contextlib import contextmanager
from typing import BinaryIO, Iterator, Any

@contextmanager
def _open_raw_source(src: str) -> Iterator[tuple[BinaryIO, int]]:
    """
    Context manager for opening a raw source (a path to a .dat file).
    Yields:
      f: file-like object opened for reading raw bytes
      size: total size in bytes of that source
    Automatically closes the file on exit.
    """
    f = None
    try:
        size = os.stat(src).st_size
        f = open(src, "rb")
        yield f, size
    finally:
        if f is not None:
            f.close()


def get_raw_info(filename, bit_depth=16, frame_size=(512, 424)):
    """
    Get info from a raw data file with specified frame dimensions and bit depth.

    Args:
    filename (str): name of raw data file
    bit_depth (int): bits per pixel (default: 16)
    frame_dims (tuple): wxh or hxw of each frame

    Returns:
    file_info (dict): dictionary containing depth file metadata
    """

    bytes_per_frame = (frame_size[0] * frame_size[1] * bit_depth) / 8

    with _open_raw_source(filename) as (_, size):
        file_info = {
            "bytes": size,
            "nframes": int(size / bytes_per_frame),
            "dims": frame_size,
            "bytes_per_frame": bytes_per_frame,
        }

    return file_info


def read_frames_raw(
    file_name: Path,
    frame_indices: list | np.ndarray | None = None,
    frame_size: tuple = (512, 424),
    movie_dtype="<u2",
) -> Iterator[np.ndarray]:
    data = np.memmap(
        file_name, dtype=np.dtype(movie_dtype), mode="r", shape=(-1, frame_size[1], frame_size[0])
    )

    if frame_indices is None:
        frame_indices = range(data.shape[0])
    
    for idx in frame_indices:
        yield data[idx]


def get_avi_metadata(path: Path) -> int:
    # TODO: tested with original depth.avi files, need to test with orbbec avi files
    with av.open(path, 'r') as container:
        video_stream = container.streams.video[0]
        # get width, height
        width = video_stream.width
        height = video_stream.height

        # try the container‐reported frame count first
        n = video_stream.frames
        if n == 0:
            print("Warning: container frame count is 0, trying to decode and count frames instead.")
            # fallback: decode & count
            n = sum(1 for _ in container.decode(video_stream))
    return {
        "nframes": n,
        "dims": (width, height),
        "fps": video_stream.average_rate,
        "bytes": width * height * 2 * n,  # each pixel is 2 bytes
    }


# simple command to pipe frames to an ffv1 file
def write_frames(
    filename,
    frames,
    threads=6,
    fps=30,
    pixel_format="gray16le",
    codec="ffv1",
    close_pipe=True,
    pipe=None,
    frame_dtype="uint16",
    slices=24,
    slicecrc=1,
    frame_size=None,
    get_cmd=False,
):
    """
    Write frames to avi file using the ffv1 lossless encoder

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    pipe (subProcess.Pipe): pipe to currently open video file.
    frame_dtype (str): indicates the data type to use when writing the videos
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)

    Returns:
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)
    if not frame_size and isinstance(frames, np.ndarray):
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and isinstance(frames, tuple):
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in tqdm(
        range(frames.shape[0]), disable=True, desc=f"Writing frames to {filename}"
    ):
        pipe.stdin.write(frames[i].astype(frame_dtype).tostring())

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


@contextmanager
def open_video_writer(filename, fps, depth_min, depth_max, cmap="jet"):
    """
    Context manager for opening a video writer.
    Yields:
      writer: function that writes frames to the video file
    Automatically closes the file on exit.
    """
    writer = iio.imopen(filename, "w", plugin="pyav")
    writer.init_video_stream(codec="h264", fps=fps, pixel_format="yuv420p")
    writer._video_stream.options = {"preset": "medium", "crf": "25"}

    cmap = plt.get_cmap(cmap)
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)

    def write_frame(frame, frame_num):
        txt_pos = (5, frame.shape[-1] - 40)

        frame = frame.astype("float32")
        frame = np.clip((frame - depth_min) / (depth_max - depth_min), 0, 1)
        frame = np.uint8(cmap(frame)[..., :3] * 255)

        frame_num_str = str(frame_num)
        try:
            cv2.putText(frame, frame_num_str, txt_pos, font, 1, white, 2, cv2.LINE_AA)
        except (IndexError, ValueError):
            # len(frame_range) M < len(frames) or txt_pos is outside of the frame dimensions
            print("Could not overlay frame number on preview on video.")

        writer.write_frame(frame)

    try:
        yield write_frame
    finally:
        writer.close()

def write_frames_preview(frames, write_fun: callable, frame_range=None):
    """
    Writes out a mp4 video where color represents height off the floor.

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    fps (int): frames per second
    depth_min (int): minimum mouse depth from floor in (mm)
    depth_max (int): maximum mouse depth from floor in (mm)
    cmap (str): color map to use.
    frame_range (range()): frame indices to write on video
    progress_bar (bool): If True, displays a TQDM progress bar for the video writing progress.
    """
    for i, frame in enumerate(frames):
        frame_num = i if frame_range is None else frame_range[i]
        write_fun(frame, frame_num)


def avi_reader(path) -> Iterator[np.ndarray]:
    with av.open(path, "r") as reader:
        reader.streams.video[0].thread_type = "AUTO"
        for frame in reader.decode(video=0):
            yield frame.to_ndarray()


def avi_video_sequence(file_path: Path, indices: np.ndarray) -> Iterator[np.ndarray]:
    with av.open(file_path) as container:
        stream = container.streams.video[0]

        for index in indices:
            # compute timestamp in seconds for the nth frame
            frame_rate = float(stream.average_rate)
            time_s = index / frame_rate

            # convert to stream.time_base units (pts)
            target_pts = int(time_s / float(stream.time_base))

            # seek to the closest keyframe at or before the target pts
            container.seek(target_pts, any_frame=False, backward=True, stream=stream)
            frame = next(container.decode(stream))
            yield frame.to_ndarray()


def indexed_video_sequence(
    file_path: Path, indices: np.ndarray, finfo: dict | None = None
) -> Iterator[np.ndarray]:
    """Selects the appropriate iterator based on the file extension."""
    if file_path.suffix == ".avi":
        return avi_video_sequence(file_path, indices)
    elif file_path.suffix == ".dat":
        if finfo is None:
            raise ValueError("dat_vid_params must be provided for .dat files")
        # dat_vid_params should contain frame_size and movie_dtype
        return read_frames_raw(file_path, indices, frame_size=finfo["dims"], movie_dtype=finfo["dtype"])


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

def batched_video_reader(
    filename: str | Path,
    n_frames: int,
    batch_size=1000,
    frame_size=(512, 424),
    bit_depth=16,
    overlap=0,
    offset=0,
    **kwargs,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:

    filename = Path(filename)

    if filename.suffix == ".avi":
        def batched_avi_reader():
            reader = avi_reader(filename)
            unique = batch_size - overlap

            buf = deque(maxlen=overlap)
            idx_buf = deque(maxlen=overlap)

            # prime the buffer
            for idx in range(offset, offset + batch_size):
                try:
                    frame = next(reader)
                except StopIteration:
                    return
                buf.append(frame)
                idx_buf.append(idx)
                yield idx, frame

            start = offset + batch_size
            # slide by unique frames
            for indices in partition_all(unique, range(start, n_frames)):

                # yield the overlapping frames first
                for i in range(overlap):
                    yield idx_buf[i], buf[i]

                # then yield the new frames, refill the buffer
                for idx in indices:
                    try:
                        frame = next(reader)
                    except StopIteration:
                        return
                    buf.append(frame)
                    idx_buf.append(idx)
                    yield idx, frame

        reader = batched_avi_reader()
    elif filename.suffix == ".dat":

        def batched_dat_reader():
            frame_batches = gen_batch_sequence(
                nframes=n_frames, chunk_size=batch_size, overlap=overlap, offset=offset
            )
            for batch in frame_batches:
                for i, frame in zip(
                    batch,
                    read_frames_raw(
                        filename, frames=batch, frame_size=frame_size, bit_depth=bit_depth, **kwargs
                    ),
                ):
                    yield i, frame

        reader = batched_dat_reader()

    for batch in partition_all(batch_size, reader):
        # batch is a list of tuples (index, frame)
        indices, frames = zip(*batch)
        yield np.array(indices), np.array(frames)


def get_movie_info(
    filename: Path, frame_size: tuple[int, int] = (512, 424), bit_depth: int = 16, **kwargs
) -> dict[str, Any]:
    """
    Return dict of movie metadata.

    Args:
    filename (Path): path to video file
    frame_size (tuple): video dimensions
    bit_depth (int): integer indicating data type encoding

    Returns:
    metadata (dict): dictionary containing video file metadata
    """

    if filename.suffix == ".dat":
        metadata = get_raw_info(
            filename, frame_size=frame_size, bit_depth=bit_depth
        )
    elif filename.suffix == ".avi":
        metadata = get_avi_metadata(filename)
    else:
        raise ValueError("Unsupported file type. Supported types are .avi and .dat")

    return metadata
