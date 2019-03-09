from typing import List, Tuple
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from .base_model import VariationalBaseModel


# Convolutional Variational Sparse Coding Model 
class ConvVSC(nn.Module):
    
    def __init__(self, input_sz: Tuple[int, int, int] = (3, 64, 64), 
                 kernel_szs: List[int] = [32, 32, 64, 64], 
                 hidden_sz: int = 256,
                 latent_sz: int = 32):
        
        super(ConvVSC, self).__init__()
        self.input_sz = input_sz 
        self.channel_szs = [input_sz[0]] + kernel_szs 
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        self.c = 50.0
        
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
#         print(self.conv_output_sz, self.flat_conv_output_sz)
        
        self.features_to_hidden = nn.Sequential(
            nn.Linear(self.flat_conv_output_sz, hidden_sz),
            nn.ReLU()
        )
        
        self.fc_mean = nn.Linear(hidden_sz, latent_sz)
        self.fc_logvar = nn.Linear(hidden_sz, latent_sz)
        self.fc_logspike = nn.Linear(hidden_sz, latent_sz)
        
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
#         print('encoder features', features.shape)
        features = features.view(-1, self.flat_conv_output_sz)
        hidden = self.features_to_hidden(features)
#         print('encoder hidden', hidden.shape)
        return self.fc_mean(hidden), self.fc_logvar(hidden), \
               -F.relu(-self.fc_logspike(hidden))

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = F.sigmoid(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)

    def decode(self, z):
        #Likelihood function
#         print('latent', z.shape)
        features = self.latent_to_features(z)
#         print('decoder features', features.shape)
        features = features.view(-1, *self.conv_output_sz)
#         print('decoder 2D features ', features.shape)
        return self.conv_decoder(features)

    def forward(self, x):
        mu, logvar, logspike = self.encode(x)
        z = self.reparameterize(mu, logvar, logspike)
        return self.decode(z), mu, logvar, logspike
    
    def update_c(self, delta):
        #Gradually increase c
        self.c += delta    

    
class ConvolutionalVariationalSparseCoding(VariationalBaseModel):
    def __init__(self, dataset, width, height, channels, kernels_szs,
                 hidden_sz, latent_sz, 
                 learning_rate, alpha, device, log_interval, normalize):
        super().__init__(dataset, width, height, channels, latent_sz,
                         learning_rate, device, log_interval, normalize)
        self.alpha = alpha
        self.hidden_sz = int(hidden_sz)
        self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]
        
        self.model = ConvVSC(self.input_sz, self.kernel_szs, self.hidden_sz,
                             latent_sz).to(device)
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
    