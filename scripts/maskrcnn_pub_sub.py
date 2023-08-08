#!/usr/bin/env python3

from mask_rcnn import MaskRCNN
import rospy
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs2
from sensor_msgs.msg import Image, CameraInfo
from utils import position2pose, project_point, img_to_cv2
import message_filters
from cv_bridge import CvBridge
import time
import tf

bridge = CvBridge()


def maskecnn_publisher(rgb_image, depth_image, camera_info):
    # 以下注释在测试时均不取消
    mask_rcnn = MaskRCNN()
    # segment_pub = rospy.Publisher(
    #     '/vision/segmentation', Image, queue_size=1)
    # pose_pub = rospy.Publisher(
    #     '/vision/pose', PoseStamped, queue_size=1)

    rgb_image = img_to_cv2(rgb_image)
    depth_image = img_to_cv2(depth_image)
    masks, boxes, labels = mask_rcnn.forward(rgb_image)
    print(labels)
    image = mask_rcnn.get_segmentation_image(
        rgb_image, masks, boxes, labels)
    target = input("pleasse input what you want to grasp:")
    target_centroid = mask_rcnn.get_target_pixel(boxes, labels, target)
    target_centroid_xyz = project_point(
        depth_image, target_centroid, camera_info)  # list
    angle = mask_rcnn.pca(masks, boxes, labels, target)  # thate

    orientation = tf.transformations.quaternion_from_euler(0, 0, angle)
    target_pose = position2pose(target_centroid_xyz, list(orientation))  # PoseStamped
    print(target_pose)
    # while not rospy.is_shutdown():
    #     segment_pub.publish(bridge.cv2_to_imgmsg(image, 'bgr8'))
    #     pose_pub.publish(target_pose)


def maskrcnn_subscriber():
    rospy.init_node('mask_rcnn_node', anonymous=True)
    rgb_image = message_filters.Subscriber(
        "/camera/color/image_raw", Image)
    depth_image = message_filters.Subscriber(
        "/camera/depth/image_rect_raw", Image)
    camera_info = message_filters.Subscriber(
        "/camera/depth/camera_info", CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer(
        [rgb_image, depth_image,
            camera_info], 10, 0.1, allow_headerless=True)
    print("Finish Sub")
    ts.registerCallback(maskecnn_publisher)
    time.sleep(3)
    rospy.spin()


if __name__ == "__main__":
    try:
        maskrcnn_subscriber()
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(e)
