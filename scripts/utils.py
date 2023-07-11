import pyrealsense2 as rs
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs2
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg

bridge = CvBridge()


def position2pose(position: list = [0, 0, 0],
                  orientation: list = [0, 0, 0, 1],
                  frame: str = "camera_color_optical_frame") -> PoseStamped:

    target_pose = PoseStamped()
    target_pose.header.frame_id = frame
    target_pose.pose.position.x = position[0]
    target_pose.pose.position.y = position[1]
    target_pose.pose.position.z = position[2]

    target_pose.pose.orientation.x = orientation[0]
    target_pose.pose.orientation.y = orientation[1]
    target_pose.pose.orientation.z = orientation[2]
    target_pose.pose.orientation.w = orientation[3]

    return target_pose


def info2intrinsic(camera_info: CameraInfo) -> rs2.intrinsics:

    intrinsics = rs2.intrinsics()
    intrinsics.width = camera_info.width
    intrinsics.height = camera_info.height
    intrinsics.ppx = camera_info.K[2]
    intrinsics.ppy = camera_info.K[5]
    intrinsics.fx = camera_info.K[0]
    intrinsics.fy = camera_info.K[4]
    intrinsics.coeffs = [i for i in camera_info.D]

    return intrinsics


def project_point(depth_image: np.ndarray,
                  xy: list,
                  camera_info: CameraInfo) -> list:
    depth = depth_image[xy[0], xy[1]] / 1000
    # convert 2D position to 3D position
    xyz = [depth * (xy[0] - camera_info.K[2]) / camera_info.K[0],
           depth * (xy[1] - camera_info.K[5]) / camera_info.K[5],
           depth]

    return xyz


def img_to_cv2(img):
    return bridge.imgmsg_to_cv2(img, img.encoding)


def test_rgbintrin_depthintrin(rgb_info: CameraInfo,
                               depth_info: CameraInfo):
    rgb_intrinsics = info2intrinsic(rgb_info)
    depth_intrinsics = info2intrinsic(depth_info)
    print(rgb_intrinsics == depth_intrinsics)


def test_intrinsics(rgb_info: CameraInfo,
                    depth_info: CameraInfo):
    print("rgb width:", rgb_info.width, "depth width:", depth_info.width,
          rgb_info.width == depth_info.width)
    print("rgb height:", rgb_info.height, "depth height:",
          depth_info.height, rgb_info.height == depth_info.height)
    print("rgb ppx:", rgb_info.K[2], "depth ppx:", depth_info.K[2],
          rgb_info.K[2] == depth_info.K[2])
    print("rgb ppy:", rgb_info.K[5], "depth ppy:", depth_info.K[5],
          rgb_info.K[5] == depth_info.K[5])
    print("rgb fx:", rgb_info.K[0], "depth fx:", depth_info.K[0],
          rgb_info.K[0] == depth_info.K[0])
    print("rgb fy:", rgb_info.K[4], "depth fy:", depth_info.K[4],
          rgb_info.K[4] == depth_info.K[4])
    print("rgb coeffs:", rgb_info.D, "depth coeffs:", depth_info.D,
          rgb_info.D == depth_info.D)


def test_project_point(camera_info: CameraInfo):
    depth = 300
    xy = [123, 321]
    xyz = project_point(depth, xy, camera_info)

    _intrinsics = info2intrinsic(camera_info)

    _xyz = rs2.rs2_deproject_pixel_to_point(_intrinsics, xy, depth)

    print(f'xyz: {xyz}\n', f'_xyz: {_xyz}\n', xyz == _xyz)


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


def gen_command(char):
    """Update the command according to the character entered by the user."""

    command = outputMsg.Robotiq2FGripper_robot_output()
    if char == 'a':
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

    if char == 'r':
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 0

    if char == 'c':
        command.rPR = 255

    if char == 'o':
        command.rPR = 0

    # If the command entered is a int, assign this value to rPRA
    try:
        command.rPR = int(char)
        if command.rPR > 255:
            command.rPR = 255
        if command.rPR < 0:
            command.rPR = 0
    except ValueError:
        pass

    if char == 'f':
        command.rSP += 25
        if command.rSP > 255:
            command.rSP = 255

    if char == 'l':
        command.rSP -= 25
        if command.rSP < 0:
            command.rSP = 0

    if char == 'i':
        command.rFR += 25
        if command.rFR > 255:
            command.rFR = 255

    if char == 'd':
        command.rFR -= 25
        if command.rFR < 0:
            command.rFR = 0

    return command


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
