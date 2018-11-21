import torch
from torch import nn, optim
from torch.nn import functional as F

from .base_model import VariationalBaseModel

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

    
class VariationalAutoEncoder(VariationalBaseModel):
    def __init__(self, dataset, width, height, hidden_sz, latent_sz, 
                 learning_rate, device, log_interval):
        super().__init__(dataset, width, height, hidden_sz, latent_sz,
                         learning_rate, device, log_interval)
        
        self.model = VAE(self.input_sz, hidden_sz, latent_sz).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    
    # Reconstruction + KL divergence losses summed over all elements of batch
    def loss_function(self, x, recon_x, mu, logvar):
        # Reconstruction term sum (mean?) per batch
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_sz), 
                                     size_average = False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
        