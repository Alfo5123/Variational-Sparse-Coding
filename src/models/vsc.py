import torch
from torch import nn, optim
from torch.nn import functional as F

from .base_model import VariationalBaseModel


# Variational Sparse Coding Model 
class VSC(nn.Module):
    def __init__(self, input_sz, hidden_sz, latent_sz):
        super(VSC, self).__init__()
        self.input_sz = input_sz # 784
        self.hidden_sz = hidden_sz # 400
        self.latent_sz = latent_sz
        self.c = 50.0
        
        self.fc1 = nn.Linear(input_sz, hidden_sz)
        self.fc21 = nn.Linear(hidden_sz, latent_sz)
        self.fc22 = nn.Linear(hidden_sz, latent_sz)
        self.fc23 = nn.Linear(hidden_sz, latent_sz)
        self.fc3 = nn.Linear(latent_sz, hidden_sz)
        self.fc4 = nn.Linear(hidden_sz, input_sz)

    def encode(self, x):
        #Recognition function
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1), -F.relu(-self.fc23(h1))

    def reparameterize(self, mu, logvar, logspike ):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = F.sigmoid(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)

    def decode(self, z):
        #Likelihood function
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar, logspike = self.encode(x.view(-1, self.input_sz))
        z = self.reparameterize(mu, logvar, logspike)
        return self.decode(z), mu, logvar, logspike
    
    def update_c(self, delta):
        #Gradually increase c
        self.c += delta
        

    
class VariationalSparseCoding(VariationalBaseModel):
    def __init__(self, dataset, width, height, hidden_sz, latent_sz, 
                 learning_rate, alpha, device, log_interval):
        super().__init__(dataset, width, height, hidden_sz, latent_sz,
                         learning_rate, device, log_interval)
        
        self.alpha = alpha
        self.model = VSC(self.input_sz, hidden_sz, latent_sz).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    
    # Reconstruction + KL divergence losses summed over all elements of batch
    def loss_function(self, x, recon_x, mu, logvar, logspike):
        # Reconstruction term sum (mean?) per batch
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_sz),
                                     size_average = False)
        # see Appendix B from VSC paper / Formula 6
        spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 
        PRIOR = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) \
                                           - logvar.exp())) + \
                       torch.sum((1 - spike).mul(torch.log((1 - spike) \
                                                /(1 - self.alpha))) + \
                       spike.mul(torch.log(spike/self.alpha)))
        return BCE + PRIOR
    
    
    def update_(self):
        # Update value of c gradually 200 ( 150 / 20K = 0.0075 )
        self.model.update_c(0.001)
    