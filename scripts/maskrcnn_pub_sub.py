# maskrcnn 订阅与发布信息
from mask_rcnn import MaskRCNN, pca
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from utils import position2pose, project_point, img_to_cv2
import message_filters
from cv_bridge import CvBridge
import time

bridge = CvBridge()


# 该回调函数用来发布消息
def maskecnn_publisher(rgb_image, depth_image, depth_image_intrinsics):
    mask_rcnn = MaskRCNN()
    segment_pub = rospy.Publisher(
        '/vision/segmentation', Image, queue_size=1)
    pose_pub = rospy.Publisher(
        '/vision/pose', PoseStamped, queue_size=1)
    rgb_image = img_to_cv2(rgb_image)
    depth_image = img_to_cv2(depth_image)
    masks, boxes, labels = mask_rcnn.forward(rgb_image)
    print(labels)
    image = mask_rcnn.get_segmentation_image(
        rgb_image, masks, boxes, labels)
    target = str(input("请输入你想抓取的目标"))
    target_centroid = mask_rcnn.get_target_pixel(boxes, labels, target)
    target_centroid_xyz = project_point(
        depth_image, target_centroid, depth_image_intrinsics)
    angle = mask_rcnn.pca(masks, boxes, labels, target)
    orientation = [target_centroid_xyz[0],
                   target_centroid_xyz[1],
                   target_centroid_xyz[2],
                   angle]
    target_pose = position2pose(target_centroid_xyz, orientation)
    while not rospy.is_shutdown():
        segment_pub.publish(bridge.cv2_to_imgmsg(image, 'bgr8'))
        pose_pub.publish(target_pose)


# maskrcnn订阅来自stream的消息
def maskrcnn_subscriber():
    rospy.init_node('mask_rcnn_node', anonymous=True)
    rgb_image = message_filters.Subscriber(
        "/camera/color/image_raw", Image)
    depth_image = message_filters.Subscriber(
        "/camera/depth/image_rect_raw", Image)
    depth_image_intrinsics = message_filters.Subscriber(
        "/camera/depth/camera_info", CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer(
        [rgb_image, depth_image,
            depth_image_intrinsics], 10, 0.1, allow_headerless=True)
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
