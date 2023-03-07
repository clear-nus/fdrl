import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, CenterCrop, Normalize
import random
import os
from PIL import Image
from sklearn import datasets
import numpy as np


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform):
        self.config = config

        # A
        images_A = os.listdir(os.path.join(config['data_dir'], config['dataset'], config['phase'] + 'A'))
        self.A = [os.path.join(config['data_dir'], config['dataset'], config['phase'] + 'A', x) for x in images_A]

        # B
        images_B = os.listdir(os.path.join(config['data_dir'], config['dataset'], config['phase'] + 'B'))
        self.B = [os.path.join(config['data_dir'], config['dataset'], config['phase'] + 'B', x) for x in images_B]

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)

        # setup image transformation
        self.transforms = transform
        return

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = self.load_img(self.A[index])
            data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)])
        else:
            data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)])
            data_B = self.load_img(self.B[index])
        return data_A, data_B

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        return img

    def __len__(self):
        return self.dataset_size


def create_dataset(config):

    if config['dataset'] == 'cifar10':
        transforms = Compose([
            Resize((config['img_size'], config['img_size'])),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        dataset = torchvision.datasets.CIFAR10(config['data_dir'], download=True, transform=transforms)

    elif config['dataset'] == 'celeba':
        transforms = Compose([
            CenterCrop(140),
            Resize((config['img_size'], config['img_size'])),
            RandomHorizontalFlip(),
            ToTensor(), 
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        dataset = torchvision.datasets.CelebA(config['data_dir'], download=True, transform=transforms)
    
    elif config['dataset'] == 'lsun_church':
        transforms = Compose([
            Resize((config['img_size'], config['img_size'])),
            CenterCrop(config['img_size']),
            RandomHorizontalFlip(),
            ToTensor(), 
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        dataset = torchvision.datasets.LSUN(config['data_dir'], transform=transforms, classes=['church_outdoor_train'])

    elif config['dataset'] == 'cat2dog':
        transforms = Compose([
            Resize((84, 84)),
            CenterCrop(config['img_size']),
            RandomHorizontalFlip(),
            ToTensor(), 
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        dataset = PairDataset(config, transform=transforms)

    else:
        print("Not a valid dataset!")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True, drop_last=False)
    return dataloader


def gaussian_dequantize(x, sigma = 3e-2):
    return x + sigma * torch.randn_like(x)


def toy_datasets(config):
    # modified from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
    BATCH_SIZE = config['train_batch_size']

    if config['dataset'] == '25gaussians':

        dataset = []
        for i in range(100000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) // BATCH_SIZE):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif config['dataset'] == 'swissroll':

        while True:
            data = datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            yield data

    elif config['dataset'] == '8gaussians':

        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset