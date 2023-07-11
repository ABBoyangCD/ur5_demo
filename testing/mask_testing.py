#!/usr/bin/env python

import message_filters
from sensor_msgs.msg import Image, CameraInfo
import random
import rospy
import string
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
import cv2

# classes names from coco
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

# make a different colour for each of the object classes
COLORS = np.random.uniform(
    0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

# a pretrained instance segmentation network (MaskRNN) that outputs the segmentation mask
# for a given image (in the form of a numpy array) and the bounding box of the
# object in the image (in the form of a numpy array)
# the model is trained on the COCO datasets

bridge = CvBridge()


class MaskRCNN:
    def __init__(self,
                 n_classes: int = 91,
                 threshold: float = 0.98):

        self.n_classes = n_classes
        self.threshold = threshold

        self.model = maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.model.eval()
        self.model.to(self.device)

        self.rgb_image = None
        self.depth_image = None

    def get_outputs(self,
                    image: np.array) -> tuple:
        # reshape from [H, W, C] to [C, H, W]
        image = image.transpose((2, 0, 1))

        # transform array to tensor of shape [batch, C, H, W] in the range [0, 1]
        image = image[np.newaxis, :, :, :] / 255.0

        # convert to torch.Tensor
        image = T.from_numpy(image).float().to(self.device)

        with T.no_grad():
            # forward pass of the image through the modle
            outputs = self.model(image)

        # for each key and value in outputs
        for key, value in outputs[0].items():
            print(key, len(value))

        # get all the scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [
            scores.index(i) for i in scores if i > self.threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        # get the masks
        masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        # discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]
        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                 for i in outputs[0]['boxes'].detach().cpu()]
        # discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]
        # get the classes labels
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i]
                  for i in outputs[0]['labels']]
        return masks, boxes, labels

    def draw_boxes(self, image, masks, boxes, labels):
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
            segmentation_map = np.stack(
                [red_map, green_map, blue_map], axis=2)
            # convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGN to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map,
                            beta, gamma, image)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                          thickness=2)
            # put the label text above the objects
            cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        thickness=2, lineType=cv2.LINE_AA)
            print(image.min(), image.max())
        return image


if __name__ == '__main__':

    # load the rgb image for testing
    rgb_image = cv2.imread("rgb_image.jpg")

    mask_rcnn = MaskRCNN()

    masks, boxes, labels = mask_rcnn.get_outputs(rgb_image)
    image = mask_rcnn.draw_boxes(rgb_image, masks, boxes, labels)

    # save the segmented image
    cv2.imwrite("segmented_image.jpg", image)
