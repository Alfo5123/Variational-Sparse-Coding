from __future__ import print_function
import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.vae import VariationalAutoEncoder


parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='input batch size for training (default: 32)')
parser.add_argument('--hidden-size', type=int, default=400, metavar='HS',
                    help='hidden size (default: 400)')
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


if __name__ == "__main__":    
    args = parser.parse_args()
    print('VAE Baseline Experiments\n')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #Set reproducibility seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #Define device for training
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'Using {device} device...')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    #Load datasets
    print(f'Loading {args.dataset} dataset...')
    if args.dataset == 'fashion':
        Dataset = datasets.FashionMNIST
        dataset_path = '../data/fashion-mnist'
        width, height = 28, 28
    else:
        Dataset = datasets.MNIST
        dataset_path = '../data/mnist'
        width, height = 28, 28

    train_loader = DataLoader(Dataset(dataset_path, train=True, download=False,
                                      transform=transforms.ToTensor()), 
                              batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(Dataset(dataset_path, train=False, download=False,
                                     transform=transforms.ToTensor()),
                             batch_size=args.batch_size, shuffle=True, **kwargs)
    print('Done!\n')
    
    # Tune the learning rate (All training rates used were between 
    # 0.001 and 0.01)
    vae = VariationalAutoEncoder('mnist', width, height, args.hidden_size, 
                                 args.latent_size, args.lr, device, 
                                 args.log_interval)
    vae.run_training(train_loader, test_loader, args.epochs,
                     args.report_interval, args.sample_size)
    
