from typing import List, Tuple
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from .base_model import VariationalBaseModel

# Convolutional Variational AutoEncoder Model 
class ConvVAE(nn.Module):
    def __init__(self, input_sz: Tuple[int, int, int] = (3, 64, 64), 
                 kernel_szs: List[int] = [32, 32, 64, 64], 
                 hidden_sz: int = 256,
                 latent_sz: int = 32,
                 beta: float = 0.1,
                 beta_delta: float = 0):
        
        super(ConvVAE, self).__init__()
        self.input_sz = input_sz 
        self.channel_szs = [input_sz[0]] + kernel_szs 
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        self.beta = beta
        self.beta_delta = beta_delta
        
        conv_modules = [(
            nn.Conv2d(self.channel_szs[i], self.channel_szs[i+1], 
                      (4, 4), stride=2, padding=1),
            nn.ReLU()
            ) for i in range(len(kernel_szs))
        ]
        
        self.conv_encoder = nn.Sequential(*[
            layer for module in conv_modules for layer in module
        ])
        
        conv_out_channels = int(input_sz[-1] / (2 ** len(kernel_szs)))
        self.conv_output_sz = (self.channel_szs[-1], conv_out_channels, 
                               conv_out_channels)
        self.flat_conv_output_sz = np.prod(self.conv_output_sz)
        
        self.features_to_hidden = nn.Sequential(
            nn.Linear(self.flat_conv_output_sz, hidden_sz),
            nn.ReLU()
        )
        
        self.fc_mean = nn.Linear(hidden_sz, latent_sz)
        self.fc_logvar = nn.Linear(hidden_sz, latent_sz)
        
        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_sz, self.hidden_sz), nn.ReLU(),
            nn.Linear(self.hidden_sz, self.flat_conv_output_sz), nn.ReLU()
        )
        
        deconv_modules = [(
            nn.ConvTranspose2d(self.channel_szs[-i-1], 
                               self.channel_szs[-i-2],
                               (4, 4), stride=2, padding=1),
            nn.ReLU() if i < len(kernel_szs) - 1 else nn.Sigmoid()
            ) for i in range(len(kernel_szs))
        ]
        
        self.conv_decoder = nn.Sequential(*[
            layer for module in deconv_modules for layer in module
        ])        
        

    def encode(self, x):
        # Recognition function
        # x shape: (batch_sz, n_channels, width)
        features = self.conv_encoder(x)
        features = features.view(-1, self.flat_conv_output_sz)
        hidden = self.features_to_hidden(features)
        return self.fc_mean(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #Likelihood function
        features = self.latent_to_features(z)
        features = features.view(-1, *self.conv_output_sz)
        return self.conv_decoder(features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def update_beta(self):
        # Gradually adjust beta
        self.beta += self.beta_delta

    
class ConvolutionalVariationalAutoEncoder(VariationalBaseModel):
    def __init__(self, dataset, width, height, channels, kernel_szs,
                 hidden_sz, latent_sz, learning_rate,
                 device, log_interval, normalize, flatten, **kwargs):
        super().__init__(dataset, width, height, channels, latent_sz,
                         learning_rate, device, log_interval, normalize, flatten)
        self.hidden_sz = int(hidden_sz)
        self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]
        
        self.model = ConvVAE(self.input_sz, self.kernel_szs, self.hidden_sz,
                             latent_sz, **kwargs).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_losses = []
        self.test_losses = []
    
    
    # Reconstruction + KL divergence losses summed over all elements of batch
    def loss_function(self, x, recon_x, mu, logvar, train=False):
        # Reconstruction term sum (mean?) per batch
        flat_input_sz = np.prod(self.input_sz)
        BCE = F.binary_cross_entropy(recon_x.view(-1, flat_input_sz), 
                                     x.view(-1, flat_input_sz),
                                     size_average = False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        LOSS = BCE + self.model.beta * KLD
        log = {
            'LOSS': LOSS.item(),
            'BCE': BCE.item(),
            'KLD': KLD.item(),
        }

        if train:
            self.train_losses.append(log)
        else:
            self.test_losses.append(log)

        return LOSS
    
    def update_(self):
        self.model.update_beta()
        