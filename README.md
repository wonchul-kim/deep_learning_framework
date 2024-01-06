# Deep Learning framework

## Classification

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | ResNet (torcivision)      | o           | o           | o           | -        |  |
|                                | Vision Transformer (ViT)  | x           | x           | x           | -        |  |
|                                | EfficientNet              | x           | x           | x           | -        |  |


## Detection

### HBB Detection

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="5">PyTorch</td>   | Yolov5                    | o           | o           | o           | -        | anchor-based                                                                |
|                                | Yolov7                    | x           | x           | x           | -        | anchor-based                                                                |
|                                | Yolov8                    | x           | x           | x           | -        | anchor-free                                                                 |
|                                | DETR                      | x           | x           | x           | -        |                                                                             |
|                                | RTDETR                    | x           | x           | x           | -        | transformer                                                                 |

### OBB Detection

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | RTMDet                    | o           | o           | o           | -        |  |
|                                |  | x           | x           | x           | -        |  |
|                                |  | x           | x           | x           | -        |  |

## Segmentation

### Semantic Segmentation

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | DeepLabV3+                | o           | o           | o           | -        |  |
|                                | Mask2Former               | x           | x           | x           | -        |  |
|                                | OneFormer                 | x           | x           | x           | -        |  |

### Instance Segmentation

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | Mask RCNN (torchvision)   | o           | o           | o           | -        |  
|                                |    | x           | -      | x           | x           | x           | -        |  
|                                |    | x           | -      | x           | x           | x           | -        |  