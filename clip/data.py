import torchvision
from torch import nn, utils
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from clip.sampler import ContrastiveSampler


def clip_transform():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_tfms = Compose([
        Resize(size=(224, 224)),
        RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    valid_tfms = Compose([
        Resize(size=(224, 224)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


    return train_tfms, valid_tfms 

def food101():
    train_tfms, valid_tfms = clip_transform()
    train_dataset = torchvision.datasets.Food101("/content/data", download=True, transform=train_tfms)
    train_sampler = ContrastiveSampler(train_dataset, shuffle=True, oversample=False)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    test_dataset = torchvision.datasets.Food101("/content/data", split= 'test', download=True, transform=valid_tfms)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader