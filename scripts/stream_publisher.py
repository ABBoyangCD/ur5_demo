#!/usr/bin/env python3

from stream import Stream
from sensor_msgs.msg import Image, CameraInfo
import rospy
from cv_bridge import CvBridge
import pyrealsense2 as rs2

bridge = CvBridge()


def stream_publisher():
    rospy.init_node('stream_node', anonymous=True)
    stream = Stream()
    stream.start()

    rgbimage_pub = rospy.Publisher("/camera/color/image_raw",
                                   Image, queue_size=10)
    depthimage_pub = rospy.Publisher("/camera/depth/image_rect_raw",
                                     Image, queue_size=10)
    intrinsic_pub = rospy.Publisher("/camera/depth/intrinsics",
                                    rs2.intrinsics(), queue_size=10)
    rate = rospy.Rate(0.1)

    while not rospy.is_shutdown():
        color_image, depth_image = stream.get_images()
        # camerainfo =  stream.get_camerainfo(stream.intrinsics)
        color_ros_image = bridge.cv2_to_imgmsg(color_image, "bgr8")
        depth_ros_image = bridge.cv2_to_imgmsg(depth_image, "passthrough")
        depth_ros_image.encoding = "16UC1"

        rgbimage_pub.publish(color_ros_image)
        depthimage_pub.publish(depth_ros_image)
        intrinsic_pub.publish(stream.intrinsics)
        # print("Finish Pub")

        rate.sleep()


if __name__ == "__main__":
    try:
        stream_publisher()
    except rospy.ROSInterruptException:
        pass
