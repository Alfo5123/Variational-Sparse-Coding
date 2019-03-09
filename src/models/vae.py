import torch
from torch import nn, optim
from torch.nn import functional as F

from .base_model import VariationalBaseModel

# Variational AutoEncoder Model 
class VAE(nn.Module):
    def __init__(self, input_sz, hidden_szs, latent_sz):
        super(VAE, self).__init__()
        self.input_sz = input_sz # 784
        self.hidden_szs = hidden_szs # [400]
        self.latent_sz = latent_sz
        
        self.fc1 = nn.Linear(input_sz, hidden_szs[0])
        self.fc1n = nn.ModuleList(
            [nn.Linear(hidden_szs[i], hidden_szs[i+1]) \
             for i in range(len(hidden_szs) - 1)])
        self.fc21 = nn.Linear(hidden_szs[-1], latent_sz)
        self.fc22 = nn.Linear(hidden_szs[-1], latent_sz)
        self.fc3 = nn.Linear(latent_sz, hidden_szs[-1])
        self.fc3n = nn.ModuleList(
            [nn.Linear(hidden_szs[-i-1], hidden_szs[-i-2]) \
             for i in range(len(hidden_szs) - 1)])
        self.fc4 = nn.Linear(hidden_szs[0], input_sz)

    def encode(self, x):
        #Recognition function
        h1 = F.relu(self.fc1(x))
        for fc in self.fc1n:
            h1 = F.relu(fc(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #Likelihood function
        h3 = F.relu(self.fc3(z))
        for fc in self.fc3n:
            h3 = F.relu(fc(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
class VariationalAutoEncoder(VariationalBaseModel):
    def __init__(self, dataset, width, height, channels, hidden_sz, latent_sz, 
                 learning_rate, device, log_interval, normalize):
        super().__init__(dataset, width, height, channels, latent_sz,
                         learning_rate, device, log_interval, normalize)
        self.hidden_sz = [int(hs) for hs in str(hidden_sz).split(',')]
        self.model = VAE(self.input_sz, self.hidden_sz, latent_sz).to(device)
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
        