from __future__ import print_function
import argparse
import torch
import os
from glob import glob
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VSC MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--latent', type=int, default=200, metavar='L',
                    help='number of latent dimensions (default: 200)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--alpha', default=0.01, type=float, metavar='A', 
                    help='value of spike variable (default: 1.0')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--fashion', action='store_true', default=False,
                    help='use fashion-mnist instead of mnist')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--report-interval', type=int, default=50, metavar='N',
                    help='how many epochs to wait before storing training status')

args = parser.parse_args()
print("VSC Baseline Experiments\n")
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#Load datasets
if args.fashion :

    print("Loading Fashion-MNIST dataset...")
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data/fashion-mnist', train=True, download=False,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data/fashion-mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print("Done!\n")

else:

    print("Loading MNIST dataset...")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=False,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print("Done!\n")


# Variational Sparse Coding Model
class VSC(nn.Module):
    def __init__(self):
        super(VSC, self).__init__()

        self.latent_dim = args.latent 
        self.c = 50.0 

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.latent_dim )
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc23 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        #Recognition function
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1), -F.relu(-self.fc23(h1))

    def reparameterize(self, mu, logvar, logspike ):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = F.sigmoid(self.c*(eta + logspike - 1))
        return selection.mul(gaussian)

    def decode(self, z):
        #Likelihood function
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar, logspike = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar, logspike)
        return self.decode(z), mu, logvar, logspike

    def update_c(self, delta):
        #Gradually increase c
        self.c += delta


#Define model 
model = VSC().to(device)

# Tune the learning rate ( All training rates used were between 0.001 and 0.01)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements of batch
def loss_function(recon_x, x, mu, logvar, logspike):

    # Reconstruction term sum (mean?) per batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average = False)

    # see Appendix B from VSC paper / Formula 6
    spyke = torch.clamp( logspike.exp() , 1e-6 , 1.0 - 1e-6 ) 
    #print(spyke)
    PRIOR = -0.5 * torch.sum( spyke.mul(1 + logvar - mu.pow(2) - logvar.exp())) + \
            torch.sum( (1-spyke).mul(torch.log((1-spyke)/(1 - args.alpha))) + \
                        spyke.mul(torch.log(spyke/args.alpha) ) )

    return BCE + PRIOR

# Run training iterations and report results
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, logspike = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, logspike)
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

# Returns the VLB for the test set
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, logspike = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, logspike).item()

    VLB = test_loss 
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} - VLB-VSC : {:.4f} '.format(test_loss, VLB))


#Auxiliary function to continue training from last model trained
def load_last_model():

    # Search for all previous checkpoints
    models = glob('models/*.pth')
    model_ids = []
    for f in models:
        if f.split('_')[0][-3:] == 'vsc':
            if args.fashion:
                if f.split('_')[1] == 'fashion':
                    model_ids.append( (int(f.split('_')[2]), f) ) 
            else:
                if f.split('_')[1] == 'mnist':
                    model_ids.append( (int(f.split('_')[2]), f) )

    # If no checkpoint available
    if len(model_ids) == 0 :
        print('Training VSC Model from scratch...')
        return 1

    # Load model from last checkpoint 
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    print('Loading VSC model from last checkpoint...')

    return start_epoch

if __name__ == "__main__":

    start_epoch = load_last_model()

    print("Training VSC model...")
    for epoch in range(start_epoch , start_epoch + args.epochs + 1):

        train(epoch)
        test(epoch)

        # Update value of c gradually 200 ( 150 / 20K = 0.0075 )
        model.update_c(0.001) 

        # For each report interval store model and save images

        if epoch % args.report_interval == 0:

            with torch.no_grad():

                ##  Generate samples
                sample = torch.randn(64, 200).to(device)
                sample = model.decode(sample).cpu()

                # Store sample plots
                if args.fashion:
                    save_image(sample.view(64, 1, 28, 28),'results/vsc_fashion_sample_' + str(epoch) + '_.png')
                else:
                    save_image(sample.view(64, 1, 28, 28),'results/vsc_mnist_sample_' + str(epoch) + '_.png')

                ## Store Model
                if args.fashion:
                    torch.save(model.cpu().state_dict(), "models/vsc_fashion_"+str(epoch)+"_.pth")
                else:
                    torch.save(model.cpu().state_dict(), "models/vsc_mnist_"+str(epoch)+"_.pth")
            
