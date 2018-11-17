from pathlib import Path
from glob import glob
from logger import Logger

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image


# Variational AutoEncoder Model 
class VAE(nn.Module):
    def __init__(self, input_sz, hidden_sz, latent_sz):
        super(VAE, self).__init__()
        self.input_sz = input_sz # 784
        self.hidden_sz = hidden_sz # 400
        self.latent_sz = latent_sz
        
        self.fc1 = nn.Linear(input_sz, hidden_sz)
        self.fc21 = nn.Linear(hidden_sz, latent_sz)
        self.fc22 = nn.Linear(hidden_sz, latent_sz)
        self.fc3 = nn.Linear(latent_sz, hidden_sz)
        self.fc4 = nn.Linear(hidden_sz, input_sz)

    def encode(self, x):
        #Recognition function
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #Likelihood function
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_sz))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
class VariationalAutoEncoder():
    def __init__(self, dataset, width, height, hidden_sz, latent_sz, 
                 learning_rate, device, log_interval):
        self.dataset = dataset
        self.width = width
        self.height = height
        self.input_sz = width * height
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        
        self.lr = learning_rate
        self.device = device
        self.log_interval = log_interval
        
        self.model = VAE(self.input_sz, hidden_sz, latent_sz).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    
    # Reconstruction + KL divergence losses summed over all elements of batch
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction term sum (mean?) per batch
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), 
                                     size_average = False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    
    def train_step(self, data):
        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.model(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
    # Run training iterations and report results
    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)            
            loss = self.train_step(data)
            train_loss += loss
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                      .format(epoch, batch_idx * len(data), 
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              loss / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        
        
    # Returns the VLB for the test set
    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu,
                                                logvar).item()
                
        VLB = test_loss / len(test_loader)
        ## Optional to normalize VLB on testset
        test_loss /= len(test_loader.dataset) 
        print('====> Test set loss: {:.4f} - VLB-VAE : {:.4f} '.format(
              test_loss, VLB))
        return test_loss
    
    
    #Auxiliary function to continue training from last trained models
    def load_last_model(self, checkpoints_path):
        # Search for all previous checkpoints
        models = glob(f'{checkpoints_path}/*.pth')
        model_ids = []
        for f in models:
            # vae_dataset_startepoch_epochs_latentsize_lr_epoch
            params = f.replace('.pth', '').split('_') 
            if params[0][-3:] == 'vae' and \
               params[1] == self.dataset and \
               int(params[4]) == self.latent_sz:
                model_ids.append((int(params[-1]), f))
                
        # If no checkpoints available
        if len(model_ids) == 0:
            print('Training VAE Model from scratch...')
            return 1

        # Load model from last checkpoint 
        start_epoch, last_checkpoint = max(model_ids, key=lambda item: item[0])
        print('Last checkpoint: ', last_checkpoint)
        self.model.load_state_dict(torch.load(last_checkpoint))
        print(f'Loading VAE model from last checkpoint ({start_epoch})...')

        return start_epoch + 1
    
    
    def run_training(self, train_loader, test_loader, epochs, 
                     report_interval, sample_sz=64,
                     checkpoints_path='../results/checkpoints',
                     logs_path='../results/logs',
                     images_path='../results/images'):
        
        start_epoch = self.load_last_model(checkpoints_path)
        run_name = f'vae_{self.dataset}_{start_epoch}_{epochs}_' \
                   f'{self.latent_sz}_{str(self.lr).replace(".", "-")}'
        logger = Logger(f'{logs_path}/{run_name}')
        print("Training VAE model...")
        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss = self.train(train_loader, epoch)
            test_loss = self.test(test_loader, epoch)
            # Store log
            logger.scalar_summary(train_loss, test_loss, epoch)
            # For each report interval store model and save images
            if epoch % report_interval == 0:
                with torch.no_grad():
                    ## Generate random samples
                    sample = torch.randn(sample_sz, self.latent_sz) \
                                  .to(self.device)
                    sample = self.model.decode(sample).cpu()
                    ## Store sample plots
                    save_image(sample.view(sample_sz, 1, self.width,
                                           self.height),
                               f'{images_path}/sample_{run_name}_{epoch}.png')
                    ## Store Model
                    torch.save(self.model.state_dict(), 
                               f'{checkpoints_path}/{run_name}_{epoch}.pth')
