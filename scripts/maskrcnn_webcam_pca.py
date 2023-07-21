from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from sklearn.decomposition import PCA


def take_photo(filename='photo.jpg', quality=1):
    js = Javascript('''
        async function takePhoto(quality) {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = 'Capture';
        div.appendChild(capture);

        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});

        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();

        // Resize the output to fit the video element.
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        // Wait for Capture to be clicked.
        await new Promise((resolve) => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        div.remove();
        return canvas.toDataURL('image/jpeg', quality);
        }
        ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


pca = PCA(n_components=2)
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
proba_threshold = 0.5
score_threshold = 0.75

try:
    filename = take_photo()
    print('Saved to {}'.format(filename))
    display(Image(filename))
except Exception as err:
    print(str(err))

image_int = read_image(str(filename))
images = [transforms(image_int)]
model = model.eval()
output = model(images)
image_output = output[0]
label = [weights.meta["categories"][label] for label in image_output['labels'][image_output['scores'] > score_threshold]]
print(label)  # 打印标签
image_bool_masks = image_output['masks'][image_output['scores'] > score_threshold] > proba_threshold
image_bool_masks = image_bool_masks.squeeze(1)
show(draw_segmentation_masks(image_int, image_bool_masks))
image_boxes = image_output["boxes"][image_output['scores'] > score_threshold]
show(draw_bounding_boxes(image_int, image_boxes, width=2))

for i in range(len(label)):
    a = image_boxes[i]
    x1, y1, x2, y2 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
    matrix = image_bool_masks[i]  # y1:y2, x1:x2
    matrix = matrix.numpy()
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
    # 计算特征向量起始点
    x_center = (x2 + x1) // 2
    y_center = (y2 + y1) // 2

    # 绘制主轴方向
    color = np.random.rand(3)
    scaled_axis = sorted_eigenvectors[0] * min(image_int.shape[1], image_int.shape[2]) * 0.1
    plt.arrow(x_center, y_center, scaled_axis[0], scaled_axis[1], color=color)
    color = np.random.rand(3)
    scaled_axis = sorted_eigenvectors[1] * min(image_int.shape[1], image_int.shape[2]) * 0.1
    plt.arrow(x_center, y_center, scaled_axis[0], scaled_axis[1], color=color)
masks = draw_segmentation_masks(image_int, image_bool_masks, alpha=0.9)
masks = masks.permute(1, 2, 0)
plt.imshow(masks.numpy());
plt.axis('off')
plt.show();