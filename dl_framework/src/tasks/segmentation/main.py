import os
import os.path as osp 
from glob import glob 

from utils.vis.vis_image import vis_image
import torch
from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
from src.penn_fudan_daaset import PennFudanDataset
from utils.augment import get_transform
from src.mask_rcnn import get_model_instance_segmentation

input_dir = '/HDD/datasets/public/PennFudanPed'
output_dir = '/HDD/outputs'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

output_dir += '/segmentation'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

# vis_image(osp.join(input_dir, 'PNGImages/FudanPed00046.png'), 
#           osp.join(input_dir, 'PedMasks/FudanPed00046_mask.png'),
#           osp.join(output_dir, 'PennFudan.png'))



# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
image_width = 640
image_height = 640
batch_size = 8
# use our dataset and defined transformations
dataset = PennFudanDataset(input_dir, get_transform(train=True, resize=(image_width, image_height)))
dataset_test = PennFudanDataset(input_dir, get_transform(train=False, resize=(image_width, image_height)))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

print("That's it!")
