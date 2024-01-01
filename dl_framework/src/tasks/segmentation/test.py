import os
import os.path as osp
import matplotlib.pyplot as plt

from torchvision.io import read_image
from utils.augment import get_transform
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from src.mask_rcnn import get_model_instance_segmentation

input_dir = '/home/wonchul/mnt/HDD/datasets/public/PennFudanPed'
output_dir = './outputs'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

    
image = read_image(osp.join(input_dir, "PNGImages/FudanPed00046.png"))
eval_transform = get_transform(train=False)


image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))