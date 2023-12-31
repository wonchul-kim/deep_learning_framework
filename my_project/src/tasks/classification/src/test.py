import warnings
import cv2 
import os.path as osp
import matplotlib.pyplot as plt 

import torch
import torch.utils.data
import utils

import torchvision
from my_project.src.tasks.classification.src.models.torchvision_model import TorchvisionModel

def test():
    model_name = 'resnet18'
    in_channels = 1
    num_classes = 10
    batch_size = 1
    workers = 8
    device = 'cuda'
    weights = '/HDD/github/deep_learning/model_9.pth'
    model = TorchvisionModel(model_name, in_channels, num_classes)
    model.load_weights(weights)
    model.to(device)
    model.eval()

    output_dir = '/HDD/github/outputs'

    dataset = torchvision.datasets.MNIST('/tmp1/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                            #    torchvision.transforms.Normalize(
                            #      (0.1307,), (0.3081,))
                             ]))
    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        sampler=sampler, num_workers=workers, pin_memory=True,
    )
    num_processed_samples = 0
    with torch.inference_mode():
        idx = 1
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)

            _image = image[0].cpu().numpy()[0, :, :]*0.3081 + 0.1307
            _image *= 255
            _label = torch.argmax(output).cpu().item()
            cv2.imwrite(osp.join(output_dir, str(idx) + '_' + str(_label) + '.png'), _image)
            idx += 1

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            num_processed_samples += batch_size
    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )


if __name__ == '__main__':
    test()