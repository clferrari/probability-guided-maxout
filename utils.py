""" helper function
"""

import sys
import numpy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def get_network(args, model_args, use_gpu=True):
    """ return given network
    """
    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(**model_args)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(**model_args)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(**model_args)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(**model_args)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(**model_args)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    if use_gpu:
        net = net.cuda()

    return net


def get_training_dataloader(dataset, mean, std, batch_size=16, num_workers=0, shuffle=True,
                            randSub=False, randSubPerc=0.5):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset == 'cifar100':
        training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == 'cifar10':
        training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == 'svhn':
        training = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    else:
        print('Wrong dataset')
        return

    if randSub:
        unique_classes = np.unique(training.targets)
        new_indices = []
        for i,c in enumerate(unique_classes):
            indices = [j for j, x in enumerate(training.targets) if x == c]
            indices = indices[:int(randSubPerc*len(indices))]
            new_indices.extend(indices)

        training_loader = DataLoader(
            training, shuffle=False, num_workers=num_workers, batch_size=batch_size,
            sampler=SubsetRandomSampler(new_indices))
    else:
        training_loader = DataLoader(
            training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader


def get_test_dataloader(dataset, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset == 'cifar100':
        test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar10':
        test = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    elif dataset == 'svhn':
        test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    else:
        print('Wrong dataset')
        return

    test_loader = DataLoader(
        test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std
