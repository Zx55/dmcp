# -*- coding:utf-8  -*-

import io
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def pil_loader(img_str):
    buff = io.BytesIO(img_str)

    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


def get_loader(config, batch_size, distributed, train_dataset, val_dataset):
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset) if distributed else None

    # Set up data loader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=(val_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


def get_cifar10(config):
    # Set up dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        config.path, transform=transform_train, train=True, download=False)

    val_dataset = datasets.CIFAR10(
        config.path, transform=transform_val, train=True, download=False)

    return train_dataset, val_dataset


def get_image_net(config):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # train
    transform_train = [
        transforms.RandomResizedCrop(config.input_size),
        transforms.RandomHorizontalFlip()
    ]

    for k in config.augmentation:
        assert k in ['test_resize', 'rotation', 'color_jitter']

    rotation = config.augmentation.get('rotation', 0)
    if rotation > 0:
        transform_train.append(transforms.RandomRotation(rotation))
    color_jitter = config.augmentation.get('color_jitter', None)
    if color_jitter is not None:
        transform_train.append(transforms.ColorJitter(*color_jitter))
    transform_train.append(transforms.ToTensor())
    transform_train.append(normalize)

    train_dir = os.path.join(config.path, 'train')
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(transform_train)
    )

    # val
    val_dir = os.path.join(config.path, 'val')
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(config.augmentation.test_resize),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            normalize
        ])
    )

    return train_dataset, val_dataset
