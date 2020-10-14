from torchvision import datasets, transforms
import torch


def load_caltech256(batch_size, test_batch_size):

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    training_set = datasets.ImageFolder('./data/caltech256/train256', transform=transform_train)
    validation_set = datasets.ImageFolder('./data/caltech256/val256', transform=transform_test)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
