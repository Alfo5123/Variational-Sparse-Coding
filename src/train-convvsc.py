import torch

from utils import get_argparser, get_datasets
from models.conv_vsc import VariationalSparseCoding

if __name__ == "__main__":    
    parser = get_argparser('ConvVSC Example')
    parser.add_argument('--alpha', default=0.5, type=float, metavar='A', 
                    help='value of spike variable (default: 0.5')
    parser.add_argument('--kernel-size', type=str, default='32,32,64,64', metavar='HS',
                        help='kernel sizes, separated by commas (default: 32,32,64,64)')
    args = parser.parse_args()
    print('ConvVSC Baseline Experiments\n')
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
    vsc = ConvolutionalVariationalSparseCoding(args.dataset, width, height, channels, 
                                  args.kernel_size, args.hidden_size, args.latent_size, 
                                  args.lr, args.alpha, device, args.log_interval,
                                  args.normalize)
    vsc.run_training(train_loader, test_loader, args.epochs,
                     args.report_interval, args.sample_size, 
                     reload_model=not args.do_not_resume)
    