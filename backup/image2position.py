import rospy
import torch
import torchvision
from torchvision.transforms import transforms as T
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray as msg_Array
from std_msgs.msg import String
import cv2
import numpy as np
import random
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

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
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))


class MaskRNN:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        self.transform = T.Compose([T.ToTensor()])

        self.image = None
        self.masks = None
        self.boxes = None
        self.labels = None

    def segment(self, image, threshold=0.965):
        self.image = image
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)

        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        thresholded_preds_inidices = [
            scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)

        self.masks = (outputs[0]['masks'] >
                      0.5).squeeze().detach().cpu().numpy()
        self.masks = self.masks[:thresholded_preds_count]
        self.boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                      for i in outputs[0]['boxes'].detach().cpu()]
        self.boxes = self.boxes[:thresholded_preds_count]
        self.labels = [COCO_INSTANCE_CATEGORY_NAMES[i]
                       for i in outputs[0]['labels']]

        return self.masks, self.boxes, self.labels

    def display(self):
        if (not self.image is None):
            alpha = 1
            beta = 0.6  # transparency for the segmentation map
            gamma = 0  # scalar added to each sum
            for i in range(len(self.masks)):
                red_map = np.zeros_like(self.masks[i]).astype(np.uint8)
                green_map = np.zeros_like(self.masks[i]).astype(np.uint8)
                blue_map = np.zeros_like(self.masks[i]).astype(np.uint8)
                # apply a randon color mask to each object
                color = COLORS[random.randrange(0, len(COLORS))]
                red_map[self.masks[i] == 1], green_map[self.masks[i]
                                                       == 1], blue_map[self.masks[i] == 1] = color
                # combine all the masks into a single image
                segmentation_map = np.stack(
                    [red_map, green_map, blue_map], axis=2)
                # convert the original PIL image into NumPy format
                image = np.array(self.image)
                # convert from RGN to OpenCV BGR format
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # apply mask on the image
                cv2.addWeighted(image, alpha, segmentation_map,
                                beta, gamma, image)
                # draw the bounding boxes around the objects
                cv2.rectangle(image, self.boxes[i][0], self.boxes[i][1], color=color,
                              thickness=2)
                # put the label text above the objects
                cv2.putText(image, self.labels[i], (self.boxes[i][0][0], self.boxes[i][0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                            thickness=2, lineType=cv2.LINE_AA)
            return image
        else:
            print('No Image')


class ImageTranslator:
    def __init__(self, rate=1,
                 rgb_image_topic='/camera/color/image_raw',
                 depth_image_topic='/camera/depth/image_rect_raw',
                 depth_info_topic='/camera/depth/camera_info'):

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        self.model = MaskRNN()
        self.label = None
        self.pix = None
        self.depth = None
        self.position = None

        self.intrinsics = None
        self.rgb_sub = rospy.Subscriber(
            rgb_image_topic, msg_Image, self.imageRGBCallback)
        self.depth_sub = rospy.Subscriber(
            depth_image_topic, msg_Image, self.imageDepthCallback)
        self.depth_info_sub = rospy.Subscriber(
            depth_info_topic, CameraInfo, self.imageDepthInfoCallback)

        self.pub = rospy.Publisher('position', msg_Array, queue_size=10)
        self.rate = rate

    def positionTalker(self):
        rospy.init_node('image2position', anonymous=True)
        while not rospy.is_shutdown():
            # msg = msg_Array()
            # msg.data = [1, 2, 3.5]
            # rospy.loginfo(msg)
            # self.pub.publish(msg)
            # rospy.Rate(self.rate).sleep()

            if (self.rgb_image is None
                or self.depth_image is None
                    or self.intrinsics is None):
                continue

            _, box, label = self.model.segment(self.rgb_image)
            if label[0] is not None:
                self.label = label[0]
                pix_x = int((box[0][0][0] + box[0][1][0]) / 2)
                pix_y = int((box[0][0][1] + box[0][1][1]) / 2)
                self.pix = [pix_x, pix_y]
                self.depth = self.depth_image[self.pix[1], self.pix[0]]

                line = f'Target object is {self.label}. '
                line += f'Central depth at pixel{self.pix}: {self.depth}. '
                self.position = rs2.rs2_deproject_pixel_to_point(
                    self.intrinsics, self.pix, self.depth)
                line += f'3D Coordinate: {self.position}.'

                pub_msg = msg_Array()
                pub_msg.data = self.position
                rospy.loginfo(line)
                self.pub.publish(pub_msg)
                rospy.Rate(self.rate).sleep()

    def imageRGBCallback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageDepthCallback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return


if __name__ == '__main__':
    # translator = ImageTranslator()
    # translator.positionTalker()
    # rospy.spin()
    try:
        translator = ImageTranslator()
        translator.positionTalker()
    except rospy.ROSInterruptException:
        pass

