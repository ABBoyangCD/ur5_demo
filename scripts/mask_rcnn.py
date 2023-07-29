import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sklearn.decomposition import PCA
import torch
import cv2
plt.rcParams["savefig.bbox"] = 'tight'


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


def calculate_angle(vector_A, vector_B):
    dot_product = np.dot(vector_A, vector_B)
    norm_A = np.linalg.norm(vector_A)
    norm_B = np.linalg.norm(vector_B)
    cos_theta = dot_product / (norm_A * norm_B)
    theta = np.arccos(cos_theta)
    # 将弧度转换为角度 Convert radians to angle
    angle_degree = np.degrees(theta)
    return angle_degree


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
        colours = [COLORS[i] for i in labels]

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

    def get_segmentation_image(self,
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
            # apply a matching colour to each mask
            color = COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(labels[i])]
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
                         target_class: str) -> tuple:
        # # Drawing bounding boxes
        # print("Found {} objects".format(len(boxes)), labels)
        try:
            # Get the index of the target class
            target_class_index = labels.index(target_class)

            target_box = boxes[target_class_index]

            # Detect target
            # Calculate 2D position of target centroid
            [x1, y1], [x2, y2] = target_box
            x1, x2 = max(x1, 0), min(x2, 639)
            y1, y2 = max(y1, 0), min(y2, 479)

            target_centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            print(f'Found {target_class} at index {target_class_index}')
            print(f'Target centroid: {target_centroid}')

            return target_centroid
        except IndexError as e:
            print(e)
            return None
        except ValueError as e:
            print(e)
            return None

    def pca(self, masks, boxes, labels, target_class):
        try:
            pca = PCA(n_components=2)
            target_class_index = labels.index(target_class)
            a = boxes[target_class_index]
            x1, y1, x2, y2 = int(a[0][0]), int(a[0][1]), int(a[1][0]), int(a[1][1])
            matrix = masks[target_class_index]
            matrix = matrix[y2:y1:-1, x1:x2]
            result = np.where(matrix.T)
            coordination = np.column_stack((result[0], result[1]))
            pca = pca.fit(coordination)
            # 取特征向量与特征值
            eigenvectors = pca.components_
            eigenvalues = pca.explained_variance_
            # 将特征向量按特征值从大到小排序
            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvectors = eigenvectors[sorted_indices, :]
            base_vector = [0, 1]
            angle = calculate_angle(base_vector, sorted_eigenvectors[0])
            print(f'Angle: {angle}')
            return angle
        except IndexError as e:
            print(e)
            return None
        except ValueError as e:
            print(e)
            return None


if __name__ == '__main__':
    image = cv2.imread('testing/images/color_image.png')
    rcnn = MaskRCNN()
    masks, boxes, labels = rcnn.forward(image)
    # print(masks)
    # print(boxes)
    print(labels)
    # segment_image = rcnn.get_segmentation_image(image, masks, boxes, labels)
    target = input("请输入目标：")
    target_centroid = rcnn.get_target_pixel(boxes, labels, target)
    angle = rcnn.pca(masks, boxes, labels, target)
    # cv2.imshow('Segmented image', segment_image)
    # cv2.waitKey(0)
    # maskrcnn_publisher()
