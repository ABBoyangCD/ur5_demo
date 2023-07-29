import numpy as np
import argparse
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
import torch
from utils import position2pose
# from mask_rcnn import MaskRCNN, COCO_INSTANCE_CATEGORY_NAMES, COLORS
import time
from stream import Stream
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import pyrealsense2 as rs
from cv_bridge import CvBridge
from mask_rcnn import MaskRCNN

bridge = CvBridge()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=["real", "sim"],
                        help='Path to image', default="real", required=False)
    parser.add_argument('-t', '--target', type=str, required=True)
    parser.add_argument('-x', '--x_offset', type=float, default=0)
    parser.add_argument('-y', '--y_offset', type=float, default=0)
    parser.add_argument('-z', '--z_offset', type=float, default=0)

    args = vars(parser.parse_args())

    rospy.init_node('vision', anonymous=True)

    rcnn = MaskRCNN()
    if args['mode'] == 'real':
        stream = Stream()
        stream.start()
    pose_pub = rospy.Publisher('/vision/pose',
                               PoseStamped, queue_size=10)
    segmentation_pub = rospy.Publisher('/vision/segmentation',
                                       Image, queue_size=10)
    try:

        while not rospy.is_shutdown():
            color_image, depth_image = stream.get_images()
            # get the masks, bounding boxes, and labels from the RCNN
            masks, bounding_boxes, labels = rcnn.forward(color_image)
            print("masks: ", len(masks), "bounding boxes: ",
                  len(bounding_boxes), "labels: ", len(labels))

            # get the segmentation Image
            segmentation_image = rcnn.get_segmentation_image(
                color_image, masks, bounding_boxes, labels)
            print("computed segmentation image")

            segmentation_pub.publish(
                bridge.cv2_to_imgmsg(segmentation_image, 'bgr8'))
            print("published segmentation image")

            # get the target pixel
            target_centroid = rcnn.get_target_pixel(
                bounding_boxes, labels, args['target'])
            print("computed target centroid")

            if target_centroid is not None:
                x, y = target_centroid
                z = depth_image[int(y), int(x)] / 1000

                # 2d position to 3d position
                position3D = rs.rs2_deproject_pixel_to_point(
                    stream.intrinsics, [x, y], z)

                position3D[0] += args['x_offset']
                position3D[1] += args['y_offset']
                position3D[2] += args['z_offset']

                print(
                    f'Target at \n\tx: {position3D[0]:.3f} y: {position3D[1]:.3f} z: {position3D[2]:.3f}')

                pose = position2pose(position3D)

                rospy.loginfo(pose)
                pose_pub.publish(pose)
                print("publishing pose")

            time.sleep(2)

    except rospy.ROSInterruptException:
        stream.stop()
        pass
    except KeyboardInterrupt:
        stream.stop()
        pass
