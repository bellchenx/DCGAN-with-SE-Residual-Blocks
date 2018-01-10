import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_loader(config):
    root = os.path.join(os.path.abspath(os.curdir), config.dataset_dir)
    print('-- Loading images')
    dataset = ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    return loader

def denorm(x):
    out = x * 0.5 + 0.5
    return out.clamp(0, 1)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('-- Total number of parameters: %d' % num_params)