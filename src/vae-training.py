import torch

from utils import get_argparser, get_datasets
from models.vae import VariationalAutoEncoder

if __name__ == "__main__":    
    parser = get_argparser('VAE Example')
    args = parser.parse_args()
    print('VAE Baseline Experiments\n')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #Set reproducibility seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #Define device for training
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'Using {device} device...')
    
    #Load datasets
    train_loader, test_loader, (width, height, channels) = get_datasets(args.dataset, 
                                                                        args.batch_size,
                                                                        args.cuda)
    
    # Tune the learning rate (All training rates used were between 0.001 and 0.01)
    vae = VariationalAutoEncoder(args.dataset, width, height, channels, 
                                 args.hidden_size, args.latent_size, args.lr, 
                                 device, args.log_interval)
    vae.run_training(train_loader, test_loader, args.epochs,
                     args.report_interval, args.sample_size)
    
