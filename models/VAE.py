import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Define some constants
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1e-8

class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()

        # The encoder network (e.g., MLP) that outputs concatenated [mu, log_var]
        self.encoder = encoder_net

    def reparameterization(self, mu, log_var):
        # Reparameterization trick:
        # Sample z ~ N(mu, sigma^2) using z = mu + sigma * epsilon,
        # where epsilon ~ N(0, I)
        std = torch.exp(0.5 * log_var)        # Convert log variance to standard deviation
        eps = torch.randn_like(std)           # Sample epsilon with the same shape as std (L=1)
        return mu + std * eps                 # Sample z

    def encode(self, x):
        # Forward input x through the encoder network to get mu and log_var
        h_e = self.encoder(x)                 # Output is of size 2M
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)  # Split into two parts: mean and log variance
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        # Return a sample from the approximate posterior
        # Can either compute mu/log_var from x or take them as input
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log_var can\'t be None!')
        z = self.reparameterization(mu_e, log_var_e)
        return z

class BernoulliDecoder(nn.Module):
    def forward(self, z):
        return self.decode(z)
    
    def __init__(self, decoder_net):
        super(BernoulliDecoder, self).__init__()
        self.decoder = decoder_net

    def decode(self, z):
        logits = self.decoder(z)
        probs = torch.sigmoid(logits)
        return probs

    def sample(self, z):
        probs = self.decode(z)
        return probs  # Return probabilities in [0,1]

    def log_prob(self, x, z):
        theta = self.decode(z)
        # BCE
        # Check x values are between 0 and 1
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError('Input x must be in the range [0, 1] for Bernoulli log_prob computation.')
        log_prob = -F.binary_cross_entropy(theta, x, reduction='none').sum(dim=1)
        return log_prob
    
class Prior(nn.Module):
    def __init__(self, M):
        super(Prior, self).__init__()

        # M is the dimensionality of the latent space
        self.M = M

    def sample(self, batch_size):
        # Sample from a standard normal distribution (mean=0, std=1)
        z = torch.randn((batch_size, self.M))  # Shape: (batch_size, latent_dim)
        return z

class VAE_Bernoulli(nn.Module):
    def __init__(self, encoder_net, decoder_net, M=16):
        super(VAE_Bernoulli, self).__init__()
        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = BernoulliDecoder(decoder_net=decoder_net)
        self.prior = Prior(M=M)

    def forward(self, x):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        RE = self.decoder.log_prob(x, z)
        KL = -0.5 * torch.sum(torch.exp(log_var_e) + mu_e**2 - 1 - log_var_e, dim=1)
        return -(RE + KL).sum()

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)