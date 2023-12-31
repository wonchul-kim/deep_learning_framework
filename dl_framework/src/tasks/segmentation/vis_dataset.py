import os
import os.path as osp
import matplotlib.pyplot as plt

import torch
from torchvision.io import read_image
from utils.augment import get_transform
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from src.penn_fudan_daaset import PennFudanDataset

input_dir = '/home/wonchul/mnt/HDD/datasets/public/PennFudanPed'
output_dir = './outputs/dataset'


image_width = 160
image_height = 160

if not osp.exists(output_dir):
    os.makedirs(output_dir)

dataset = PennFudanDataset(input_dir, get_transform(train=False, resize=(image_width, image_height)))

for image, target, img_path in dataset:  
    filename = osp.split(osp.splitext(img_path)[0])[-1]
    pred_labels = [f"pedestrian" for label in zip(target["labels"])]
    pred_boxes = target["boxes"].long()
    output_image = draw_bounding_boxes(image.to(torch.uint8), pred_boxes, pred_labels, colors="red")

    masks = (target["masks"] > 0.7).squeeze(1)
    # masks = target['masks'].squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig(osp.join(output_dir, filename + '.png'))