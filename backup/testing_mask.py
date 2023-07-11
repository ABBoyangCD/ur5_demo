#!/usr/bin/env python

import time
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
import rospy
import pyrealsense2 as rs2
from src.ur5_demo.backup.testing_mask import MaskRCNN





def make_intrinsic(camera_info: CameraInfo) -> rs2.intrinsics:
    intrinsics = rs2.intrinsics()
    intrinsics.width = camera_info.width
    intrinsics.height = camera_info.height
    intrinsics.ppx = camera_info.K[2]
    intrinsics.ppy = camera_info.K[5]
    intrinsics.fx = camera_info.K[0]
    intrinsics.fy = camera_info.K[4]
    intrinsics.coeffs = [i for i in camera_info.D]
    return intrinsics

def _test_project_point(camera_info: CameraInfo):
    depth = 300
    xy = [123, 321]
    xyz = self.project_point(depth, xy, camera_info)

    _intrinsics = self.make_intrinsic(camera_info)

    _xyz = rs2.rs2_deproject_pixel_to_point(_intrinsics, xy, depth)

    print(f'xyz: {xyz}\n', f'_xyz: {_xyz}\n', xyz == _xyz)

def _test_rgbintrin_depthintrin(self,
                                rgb_info: CameraInfo,
                                depth_info: CameraInfo):
    rgb_intrinsics = self.make_intrinsic(rgb_info)
    depth_intrinsics = self.make_intrinsic(depth_info)
    print(rgb_intrinsics == depth_intrinsics)

def _test_intrinsics(self,
                     rgb_info: CameraInfo,
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


if __name__ == '__main__':
    mask_rcnn = MaskRCNN()
    segment_pub = rospy.Publisher(
        '/mask_rcnn/segment_image', Image, queue_size=1)
    target_pub = rospy.Publisher(
        '/mask_rcnn/target_position', PoseStamped, queue_size=1)
    pointcloud_pub = rospy.Publisher(
        '/mask_rcnn/pointcloud', PointCloud2, queue_size=1)

    def callback(rgb_image, rgb_image_intrinsics, depth_image, depth_image_intrinsics, pointcloud):
        rgb_image = bridge.imgmsg_to_cv2(rgb_image, rgb_image.encoding)
        depth_image = bridge.imgmsg_to_cv2(depth_image, depth_image.encoding)

        masks, boxes, labels = mask_rcnn.forward(rgb_image)
        image = mask_rcnn.segment(rgb_image, masks, boxes, labels)
        target_centroid = mask_rcnn.get_target_pixel(boxes, labels)
        target_centroid_xyz = mask_rcnn.project_point(
            depth_image, target_centroid, depth_image_intrinsics)
        target_pose = mask_rcnn.xyz_to_pose(target_centroid_xyz)

        segment_pub.publish(bridge.cv2_to_imgmsg(image, 'bgr8'))
        target_pub.publish(target_pose)
        pointcloud_pub.publish(pointcloud)

        print(image.shape)

    try:
        rospy.init_node('mask_rcnn_node', anonymous=True)
        rgb_image = message_filters.Subscriber(
            "/camera/color/image_raw", Image)
        rgb_image_intrinsics = message_filters.Subscriber(
            "/camera/color/camera_info", CameraInfo)
        depth_image = message_filters.Subscriber(
            "/camera/depth/image_rect_raw", Image)
        depth_image_intrinsics = message_filters.Subscriber(
            "/camera/depth/camera_info", CameraInfo)
        point_cloud = message_filters.Subscriber(
            "/camera/depth/color/points", PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_image, rgb_image_intrinsics, depth_image,
                depth_image_intrinsics, point_cloud], 1, 0.1, allow_headerless=True)
        ts.registerCallback(callback)
        time.sleep(5)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(e)
