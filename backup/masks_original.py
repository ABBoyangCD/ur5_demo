#!/usr/bin/env python

import cv2
import time
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
import random
import rospy
from cv_bridge import CvBridge
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
import pyrealsense2 as rs2
import torch
from torchvision import transforms as T

# Classes names from coco
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Make a different colour for each of the object classes
COLORS = np.random.uniform(
    0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))


bridge = CvBridge()


class MaskRCNN:
    def __init__(self,
                 threshold: float = 0.92):
        # Set threshold score and detection target
        self.threshold = threshold

        # Initialize detection network
        self.model = maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        self.model.to(self.device)

        # Convert from numpy array (H x W x C) in the range [0, 255]
        #   to tensor (C x H x W) in the range [0.0, 1.0]
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def forward(self,
                image: np.ndarray,
                print_results: bool = False):

        # Transform the image
        image = self.transform(image)
        # Add a batch dimension
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Forward pass of the image through the model
            outputs = self.model(image)[0]

        # Get all the scores
        scores = list(outputs['scores'].detach().cpu().numpy())
        # Index of those scores which are above a certain threshold
        thresholded_preds_inidices = [
            scores.index(i) for i in scores if i > self.threshold]

        thresholded_preds_count = len(thresholded_preds_inidices)

        # Get the masks
        masks = (outputs['masks'] > 0.5).squeeze().detach().cpu().numpy()
        # Discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]
        # Get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                 for i in outputs['boxes'].detach().cpu()]
        # Discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]

        # get labels that pass the treshold and are in the list of classes
        labels = outputs['labels'][:thresholded_preds_count]

        # Get the classes labels
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i]
                  for i in labels]
        if print_results:
            # Print results as 'Label: Score'
            scores = scores[:thresholded_preds_count]
            results = ''
            for i in range(thresholded_preds_count):
                results += f'{labels[i]}: {scores[i]} '
            print(results)

        return masks, boxes, labels

    def segment(self,
                image: np.array,
                masks:  list,
                boxes: list,
                labels: list) -> np.array:
        alpha = 1
        beta = 0.6  # transparency for the segmentation map
        gamma = 0  # scalar added to each sum
        for i in range(len(masks)):
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # apply a randon color mask to each object
            color = COLORS[random.randrange(0, len(COLORS))]
            red_map[masks[i] == 1], green_map[masks[i]
                                              == 1], blue_map[masks[i] == 1] = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            # convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGN to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                          thickness=2)
            # put the label text above the objects
            cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        thickness=2, lineType=cv2.LINE_AA)
        return image

    def get_target_pixel(self,
                         boxes: list,
                         labels: list,
                         target_class: str = 'keyboard') -> tuple:
        # Drawing bounding boxes
        print("Found {} objects".format(len(boxes)), labels)

        try:
            # Get the index of the target class
            target_class_index = labels.index(target_class)
            print(f'Found {target_class} at index {target_class_index}')

            target_box = boxes[target_class_index]

            # Calculate 2D position of target centroid
            [x1, y1], [x2, y2] = target_box
            x1, x2 = max(x1, 0), min(x2, 639)
            y1, y2 = max(y1, 0), min(y2, 479)

            target_centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            return target_centroid
        except IndexError as e:
            print(e)
            return None
        except ValueError:
            print(f'{target_class.capitalize()} not found in labels')
            return None

    # convert 2D centroid and depth to 3D position

    # converts 3D position to PoseStamped message

    def xyz_to_pose(self,
                    xyz: list,
                    frame: str = "camera_color_optical_frame") -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]
        return pose

    # projects a point given the depth, 2D position and camera parameters
    def project_point(self,
                      depth: float,
                      xy: list,
                      camera_info: CameraInfo) -> list:
        # convert 2D position to 3D position
        xyz = [depth * (xy[0] - camera_info.K[2]) / camera_info.K[0],
               depth * (xy[1] - camera_info.K[5]) / camera_info.K[4],
               depth]
        return xyz

    @staticmethod
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

    # Image to cv2 Image
    @staticmethod
    def image_to_cv2(image: Image) -> np.array:
        cv2_image = bridge.imgmsg_to_cv2(image, "bgr8")
        return cv2_image

    def _test_project_point(self, camera_info: CameraInfo):
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
        rgb_image = image_to_cv2(rgb_image)
        depth_image = image_to_cv2(depth_image)

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
