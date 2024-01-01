import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch import optim

from model.vit import ViT

from sklearn.metrics import confusion_matrix, accuracy_score
from src.dataset import get_loader


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = get_loader(args)

        self.model = ViT(in_channels=1).cuda()
        self.ce = nn.CrossEntropyLoss()

        print('--------Network--------')
        print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'Transformer.pt')))

    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []

        if db.lower() == 'train':
            loader = self.train_loader
        else:
            loader = self.test_loader

        for (imgs, labels) in loader:
            imgs = imgs.cuda()

            with torch.no_grad():
                class_out = self.model(imgs)
            _, predicted = torch.max(class_out.data, 1)

            actual += labels.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

        return acc, cm

    def test(self):
        train_acc, cm = self.test_dataset('train')
        print("Tr Acc: %.2f" % train_acc)
        print(cm)

        test_acc, cm = self.test_dataset('test')
        print("Te Acc: %.2f" % test_acc)
        print(cm)

        return train_acc, test_acc

    def train(self):
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs, verbose=True)

        for epoch in range(self.args.epochs):

            self.model.train()

            for i, (imgs, labels) in enumerate(self.train_loader):

                imgs, labels = imgs.cuda(), labels.cuda()

                logits = self.model(imgs)
                clf_loss = self.ce(logits, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (epoch + 1, self.args.epochs, i + 1, iter_per_epoch, clf_loss))

            test_acc, cm = self.test_dataset('test')
            print("Test acc: %0.2f" % test_acc)
            print(cm, "\n")

            cos_decay.step()

import argparse
import datetime
import os

def main(args):
    os.makedirs(args.model_path, exist_ok=True)

    solver = Solver(args)
    solver.train()
    solver.test()


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--log_step', type=int, default=50)

    parser.add_argument('--dset', type=str, default='mnist', help=['mnist', 'fmnist'])
    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels")
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='./model')

    parser.add_argument("--embed_dim", type=int, default=96, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=4, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2, help="forward multiplier")
    parser.add_argument("--n_layers", type=int, default=6, help="number of encoder layers")
    parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, args.dset)
    print(args)
    
    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))