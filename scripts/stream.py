import numpy as np
import time
import pyrealsense2 as rs

import cv2


class Stream:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Our device is D415, both resolutions are 640 x 480 (W x H)
        # Set stream type, resolution, format, frame rate
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Intrinsics to be used for calculating 3D position
        self.intrinsics = None

    def start(self):
        # Start streaming with configuration
        pipeline_profile = self.pipeline.start(self.config)
        # Fetch stream profile for depth stream
        depth_profile = pipeline_profile.get_stream(rs.stream.depth)
        # Downcast to video_stream_profile and fetch intrinsics
        self.intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    def stop(self):
        # Important! Write in the finally block to make sure always called
        # Stop streaming and release the device resources used by the pipeline
        self.pipeline.stop()

    def get_images(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        # Retrieve the first depth and color frame of the fetched frames sets
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print('Failed to fetch frames set!')
            return

        # Convert images to numpy arrays (H x W x C)
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image


if __name__ == '__main__':

    try:
        stream = Stream()
        stream.start()
        while True:
            color_image, depth_image = stream.get_images()

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(
                    depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.imwrite(
                './src/ur5_demo/testing/images/color_image.png', color_image)
            cv2.waitKey(1)
            time.sleep(1)

    finally:
        stream.stop()
