import av
import numpy as np

# Video parameters
width = 256
height = 256
fps = 30
total_frames = 100

# Output file
output_file = 'output_16bit_gray.avi'

# Open the output file for writing
with av.open(output_file, mode='w') as container:
    # Add a video stream
    stream = container.add_stream('ffv1', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'gray16le'  # 16-bit little-endian grayscale

    # Generate and write frames
    for frame_i in range(total_frames):
        # Create a 16-bit grayscale frame (example: ramp from black to white)
        gray_value = int(frame_i / total_frames * 65535)  # Scale to 16-bit range
        frame_data = np.full((height, width), gray_value, dtype=np.uint16)

        # Create a PyAV frame from the numpy array
        frame = av.VideoFrame.from_ndarray(frame_data, format='gray16le')
        # frame.pts = int(frame_i / fps * 1000000)  # Calculate timestamp in microseconds

        # Encode and write the frame
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush the encoder
    for packet in stream.encode():
        container.mux(packet)
