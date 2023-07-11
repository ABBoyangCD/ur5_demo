#!/usr/bin/env python

from utils import position2pose, project_point, img_to_cv2
from mask_rcnn import MaskRCNN
import time
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
import rospy
from cv_bridge import CvBridge


bridge = CvBridge()


if __name__ == '__main__':
    mask_rcnn = MaskRCNN()
    segment_pub = rospy.Publisher(
        '/mask_rcnn/segment_image', Image, queue_size=1)
    target_pub = rospy.Publisher(
        '/mask_rcnn/target_position', PoseStamped, queue_size=1)
    pointcloud_pub = rospy.Publisher(
        '/mask_rcnn/pointcloud', PointCloud2, queue_size=1)

    def callback(rgb_image, depth_image, depth_image_intrinsics, pointcloud):
        rgb_image = img_to_cv2(rgb_image)
        depth_image = img_to_cv2(depth_image)

        masks, boxes, labels = mask_rcnn.forward(rgb_image)
        image = mask_rcnn.get_segmentation_image(
            rgb_image, masks, boxes, labels)

        target_centroid = mask_rcnn.get_target_pixel(boxes, labels)

        target_centroid_xyz = mask_rcnn.project_point(
            depth_image, target_centroid, depth_image_intrinsics)

        target_pose = position2pose(target_centroid_xyz)

        segment_pub.publish(bridge.cv2_to_imgmsg(image, 'bgr8'))
        target_pub.publish(target_pose)
        pointcloud_pub.publish(pointcloud)

        print("x:", target_centroid_xyz[0], "y:",
              target_centroid_xyz[1], "z:", target_centroid_xyz[2])

    try:
        rospy.init_node('mask_rcnn_node', anonymous=True)
        rgb_image = message_filters.Subscriber(
            "/camera/color/image_raw", Image)
        depth_image = message_filters.Subscriber(
            "/camera/depth/image_rect_raw", Image)
        depth_image_intrinsics = message_filters.Subscriber(
            "/camera/depth/camera_info", CameraInfo)
        point_cloud = message_filters.Subscriber(
            "/camera/depth/color/points", PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_image, depth_image,
                depth_image_intrinsics, point_cloud], 10, 0.1, allow_headerless=True)
        ts.registerCallback(callback)
        time.sleep(3)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(e)
