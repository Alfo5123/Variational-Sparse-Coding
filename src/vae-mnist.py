from __future__ import print_function
import os
import torch
import argparse
from glob import glob
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from logger import Logger


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--latent', type=int, default=200, metavar='L',
                    help='number of latent dimensions (default: 200)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=11, metavar='N',
                    help='number of epochs to train (default: 11)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--fashion', action='store_true', default=False,
                    help='use fashion-mnist instead of mnist')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--report-interval', type=int, default=11, metavar='N',
                    help='how many epochs to wait before storing training status')

args = parser.parse_args()
print("VAE Baseline Experiments\n")
args.cuda = not args.no_cuda and torch.cuda.is_available()

#Set reproducibility seeed
torch.manual_seed(args.seed)

#Define device for training
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#Load datasets
if args.fashion :

    print("Loading Fashion-MNIST dataset...")
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data/fashion-mnist', train=True, download=False,
        transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data/fashion-mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print("Done!\n")

else:

    print("Loading MNIST dataset...")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=False,
        transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print("Done!\n")


# Variational AutoEncoder Model 
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, args.latent)
        self.fc22 = nn.Linear(400, args.latent)
        self.fc3 = nn.Linear(args.latent, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        #Recognition function
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #Likelihood function
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


#Define model 
model = VAE().to(device)

# Tune the learning rate ( All training rates used were between 0.001 and 0.01)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements of batch
def loss_function(recon_x, x, mu, logvar):

    # Reconstruction term sum (mean?) per batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average = False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Run training iterations and report results
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return train_loss / len(train_loader.dataset) 

# Returns the VLB for the test set
def test(epoch):

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    VLB = test_loss / ( i +  1) # Average loss per batch
    test_loss /= len(test_loader.dataset) # Loss per number of observations
    print('====> Test set loss: {:.4f} - VLB-VAE : {:.4f} '.format(test_loss, VLB))

    return test_loss

#Auxiliary function to continue training from last model trained
def load_last_model():

    # Search for all previous checkpoints
    models = glob('models/*.pth')
    model_ids = []
    for f in models:
        if f.split('_')[0][-3:] == 'vae':
            if args.fashion:
                if f.split('_')[1] == 'fashion':
                    if args.latent == int(f.split('_')[-4]):
                        model_ids.append( (int(f.split('_')[-2]), f) ) 
            else:
                if f.split('_')[1] == 'mnist':
                    if args.latent == int(f.split('_')[-4]):
                        model_ids.append( (int(f.split('_')[-2]), f) )

    # If no checkpoint available
    if len(model_ids) == 0 :
        print('Training VAE Model from scratch...')
        return 1

    # Load model from last checkpoint 
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    print('Loading VAE model from last checkpoint...')

    return start_epoch+1

if __name__ == "__main__":

    #Load model weights from latest checkpoint
    #start_epoch = 1
    start_epoch = load_last_model()

    # Store log characteristic name for each run
    ## model_dataset_startepoch_numepochs_latent_lr
    if args.fashion:
        run_name = 'vae_fashion_' + str(start_epoch) + '_' + str(args.epochs) + '_' + \
                    str(args.latent) + '_' + str(args.lr).replace('.','-')
    else:
        run_name = 'vae_mnist_' + str(start_epoch) + '_' + str(args.epochs) + '_' + \
                    str(args.latent)  + '_' + str(args.lr).replace('.','-')
    
    logger = Logger('./logs' + '/' + run_name )

    print("Training VAE model...")
    for epoch in range(start_epoch , start_epoch + args.epochs ):

        train_loss = train(epoch)
        test_loss = test(epoch)

        # Store log
        logger.scalar_summary(train_loss, test_loss, epoch)
        
        # For each report interval store model and save images
        if epoch % args.report_interval == 0:

            with torch.no_grad():

                ## Generate random samples
                sample = torch.randn(64, args.latent).to(device)
                sample = model.decode(sample).cpu()

                ## Store sample plots
                save_image(sample.view(64, 1, 28, 28),'results/sample_' + run_name + '_' + str(epoch) + '_.png')

                ## Store Model
                torch.save(model.cpu().state_dict(), "models/" + run_name + '_' + str(epoch) + "_.pth")
            
