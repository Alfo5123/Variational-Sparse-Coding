from __future__ import print_function
import argparse
import torch
from glob import glob
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
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
print("VAE Baseline Experiments\n")
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



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 200)
        self.fc22 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


#Define model 
model = VAE().to(device)

# Tune the learning rate ( All training rates used were between 0.001 and 0.01)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

    # Reconstruction term sum (mean?) per batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average = False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


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


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            ## Plot reconstruction comparison
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])

                if args.fashion:
                    save_image(comparison.cpu(),
                             'results/reconstruction_fashion' + str(epoch) + '.png', nrow=n)
                else:
                    save_image(comparison.cpu(),
                         'results/reconstruction_mnist' + str(epoch) + '.png', nrow=n)



    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def load_last_model():

    # Auxiliary function to continue training from last model trained

    models = glob('models/*.pth')
    model_ids = [(int(f.split('_')[2]), f) for f in models]

    # If no pretrained model 
    if len(model_ids) == 0 :
        print('Training Model from scratch...')
        return 1, -1

    # Load model from last checkpoint 
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    print('Loading model from last checkpoint...')

    return start_epoch, last_cp

if __name__ == "__main__":

    start_epoch, last_cp = load_last_model()

    print("Training model...")
    for epoch in range(start_epoch , start_epoch + args.epochs + 1):

        train(epoch)
        test(epoch)

        # For each report interval store model and save images

        if epoch % args.report_interval == 0:

            with torch.no_grad():

                ## Plot Samples
                sample = torch.randn(64, 200).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28),'results/sample_' + str(epoch) + '_.png')

                ## Store Model
                if args.fashion:
                    torch.save(model.cpu().state_dict(), "models/vae_fashion_"+str(epoch)+"_.pth")
                else:
                    torch.save(model.cpu().state_dict(), "models/vae_mnist_"+str(epoch)+"_.pth")
            
