#!/usr/bin/env python3

from stream import Stream
from sensor_msgs.msg import Image, CameraInfo
import rospy


def stream_publisher(color_image):
    stream = Stream
    color_image, depth_image = stream.get_images()
    stream.start()
    camerainfo = stream.get_camerainfo(stream.intrinsics)
    rospy.init_node("stream_node", anonymous=True)
    rgbimage_pub = rospy.Publisher("/camera/color/image_raw",
                                   Image, queue_size=10)
    depthimage_pub = rospy.Publisher("/camera/depth/image_rect_raw",
                                     Image, queue_size=10)
    depthimageintrinsics_pub = rospy.Publisher("/camera/depth/camera_info",
                                               CameraInfo, queue_size=10)
    while not rospy.is_shutdown():
        rgbimage_pub.publish(color_image)
        depthimage_pub.publish(depth_image)
        depthimageintrinsics_pub.publish(camerainfo)


if __name__ == "__main__":
    try:
        stream_publisher()
    except rospy.ROSInterruptException:
        pass
