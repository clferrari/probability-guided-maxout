
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets_utils import load_caltech256
from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader


def train(epoch):

    train_loss = 0.0
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()

        if args.exp == 'adaptDrop':
            outputs, feat = net(images, labels)
            loss = loss_function(outputs, labels)
            loss = loss.mean()
        else:
            outputs, feat = net(images, labels)
            loss = loss_function(outputs, labels)
            loss = loss.mean()

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

    writer.add_scalar('Train/loss', train_loss, epoch)
    if args.exp == 'adaptDrop':
        writer.add_scalar('Train/scale', net.normactive.scale, epoch)


def eval_training(epoch):
    net.eval()

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        if args.exp == 'adaptDrop':
            outputs, feat = net(images, labels)
            loss = loss_function(outputs, labels)
        else:
            outputs, feat = net(images, labels)
            loss = loss_function(outputs, labels)

        test_loss += loss.mean().item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
    print()

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / (len(test_loader.dataset)/args.b), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Top1-Error', 1 - (correct.float() / len(test_loader.dataset)), epoch)

    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':

    datasets = ['cifar10', 'cifar100', 'svhn', 'caltech256']
    expSetups = ['baseline', 'adaptDrop', 'dropout']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-exp', default='adaptDrop', choices=expSetups)
    parser.add_argument('-logDir', default='DEBUG')
    parser.add_argument('-dataset', default='cifar100', choices=datasets)
    parser.add_argument('-lossRed', default='none')
    parser.add_argument('-resume', default=False)
    parser.add_argument('-resumeNet', default='')
    parser.add_argument('-randSub', default=False)
    parser.add_argument('-randSubPerc', default=0.25)

    args = parser.parse_args()

    model_args = {'exp': args.exp}

    if args.dataset == 'caltech256':
        args.b = 32
        training_loader, test_loader = load_caltech256(args.b, args.b)
        model_args['num_classes'] = 257
    else:
        if args.dataset == 'cifar100':
            dataset_mean = settings.CIFAR100_TRAIN_MEAN
            dataset_std = settings.CIFAR100_TRAIN_STD
            model_args['num_classes'] = 100
        elif args.dataset == 'cifar10':
            dataset_mean = settings.CIFAR10_TRAIN_MEAN
            dataset_std = settings.CIFAR10_TRAIN_STD
            model_args['num_classes'] = 10
        elif args.dataset == 'svhn':
            dataset_mean = settings.SVHN_TRAIN_MEAN
            dataset_std = settings.SVHN_TRAIN_STD
            model_args['num_classes'] = 10

        # data preprocessing:
        training_loader = get_training_dataloader(
            args.dataset,
            dataset_mean,
            dataset_mean,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s,
            randSub = args.randSub,
            randSubPerc=args.randSubPerc
        )

        test_loader = get_test_dataloader(
            args.dataset,
            dataset_mean,
            dataset_mean,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s
        )

    net = get_network(args, model_args, use_gpu=args.gpu)

    loss_function = nn.CrossEntropyLoss(reduction='none')

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.dataset, args.net, args.logDir)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    start_epoch = 1
    end_epoch = settings.EPOCH

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    if args.resume:
        print('Resuming form checkpoint')
        checkpoint = torch.load(os.path.join(settings.CHECKPOINT_PATH, args.dataset, args.net, args.resumeNet))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.dataset, args.net, args.logDir + '-' + settings.TIME_NOW))
        end_epoch = settings.EPOCH + 100

    else:
        writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.dataset, args.net, args.logDir + '-' + settings.TIME_NOW))

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(start_epoch, end_epoch):

        train(epoch)
        acc = eval_training(epoch)

        if best_acc < acc:
            best_acc = acc

        if epoch == settings.SAVE_EPOCH:
            torch.save({'model_state_dict': net.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_path.format(net=args.net, epoch=epoch, type='intermediate'))

    torch.save({'model_state_dict': net.state_dict(),
                'epoch': settings.EPOCH,
                'optimizer_state_dict': optimizer.state_dict()},
               checkpoint_path.format(net=args.net, epoch=settings.EPOCH, type='end'))

    writer.close()
