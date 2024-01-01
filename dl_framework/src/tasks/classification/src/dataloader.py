import torch

from torch.utils.data.dataloader import default_collate
from dl_framework.src.data.augment.transforms import get_mixup_cutmix

def get_dataloader(args, dataset, train_sampler, dataset_test, test_sampler):

    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, 
        sampler=test_sampler, num_workers=args.workers, pin_memory=True,
    )

    return data_loader, data_loader_test