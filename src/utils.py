import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import CelebA


def get_argparser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 32)')
    # Hidden size for CelebA: 2000 dimensions, 2 layers
    parser.add_argument('--hidden-size', type=int, default=400, metavar='HS',
                        help='hidden size (default: 400)')
    # Latent size for CelebA: 800 dimensions
    parser.add_argument('--latent-size', type=int, default=200, metavar='LS',
                        help='number of latent dimensions (default: 200)')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=11, metavar='N',
                        help='number of epochs to train (default: 11)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dataset', default='mnist',
                        help='dataset [mnist, fashion, celeba]')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='LOG',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--report-interval', type=int, default=11, metavar='REP',
                        help='how many epochs to wait before storing training status')
    parser.add_argument('--sample-size', type=int, default=64, metavar='SS',
                        help='how many images to include in sample image')
    return parser


def get_datasets(dataset, batch_size, cuda):
    #Load datasets
    print(f'Loading {dataset} dataset...')
    if dataset == 'fashion':
        Dataset = datasets.FashionMNIST
        dataset_path = '../data/fashion-mnist'
        width, height = 28, 28
    elif dataset == 'mnist':
        Dataset = datasets.MNIST
        dataset_path = '../data/mnist'
        width, height = 28, 28
    elif dataset == 'celeba':
        Dataset = CelebA
        dataset_path = '../data/celeba'
        width, height = 32, 32
    else:
        raise ValueError('Dataset not supported')

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        
    train_loader = DataLoader(Dataset(dataset_path, train=True, download=False,
                                      transform=transforms.ToTensor()), 
                              batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(Dataset(dataset_path, train=False, download=False,
                                     transform=transforms.ToTensor()),
                             batch_size=batch_size, shuffle=True, **kwargs)
    print('Done!\n')
    return train_loader, test_loader, (width, height)