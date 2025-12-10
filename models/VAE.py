import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.conditional_utils import get_condition


# Define some constants
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1e-8

############### VAE with Bernoulli decoder ###############
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
        z = mu + std * eps                     # Sample z
        return z                # Sample z

    def forward(self, x):
        # Forward input x through the encoder network to get mu and log_var
        h_e = self.encoder(x)                 # Output is of size 2M
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)  # Split into two parts: mean and log variance
        log_var_e = torch.clamp(log_var_e, min=-6.0, max=6.0) # Clamp log_var for numerical stability
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        # Return a sample from the approximate posterior
        # Can either compute mu/log_var from x or take them as input
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.forward(x)
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

    def forward(self, z):
        logits = self.decoder(z)
        probs = torch.sigmoid(logits)
        return probs

    def sample(self, z):
        probs = self.forward(z)
        return probs  # Return probabilities in [0,1]

    def log_prob(self, x, z):
        theta = self.forward(z)  # Use sigmoid output
        # Check x values are between 0 and 1
        if torch.any(theta < 0) or torch.any(theta > 1) or torch.isnan(theta).any():
            raise ValueError(f"[ERROR] theta (decoder output) out of bounds: min={theta.min()}, max={theta.max()}")
        if torch.any(x < 0) or torch.any(x > 1) or torch.isnan(x).any():
            raise ValueError('Input x must be in the range [0, 1] for Bernoulli log_prob computation.')
        # BCE
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
        mu_e, log_var_e = self.encoder.forward(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        RE = self.decoder.log_prob(x, z)
        KL = -0.5 * torch.sum(torch.exp(log_var_e) + mu_e**2 - 1 - log_var_e, dim=1)
        NLL = -(RE + KL).mean()
        return NLL, KL.mean()

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)


################ Conditional VAE ###############
class ConditionalEncoder(nn.Module):
    def __init__(self, encoder_net, y_species_dim, y_embed_dim, y_amr_dim, embedding=True):
        super().__init__()
        self.encoder = encoder_net
        self.y_embed = nn.Embedding(y_species_dim, y_embed_dim)
        self.y_amr_dim = y_amr_dim
        self.embedding = embedding

        self.input_dim = y_embed_dim if embedding else y_species_dim

    def get_cond(self, y_species, y_amr):
        return get_condition(y_species, self.input_dim, y_amr, self.y_embed, self.y_amr_dim, embedding=self.embedding)

    def forward(self, x, y_species, y_amr=None):
        cond = self.get_cond(y_species, y_amr)
        h_e = self.encoder(x, cond)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, y_species=None, y_amr=None, mu_e=None, log_var_e=None):
        if (mu_e is None) or (log_var_e is None):
            mu_e, log_var_e = self.forward(x, y_species, y_amr)
        std = torch.exp(0.5 * log_var_e)
        eps = torch.randn_like(std)
        return mu_e + std * eps

class ConditionalDecoder(nn.Module):
    def __init__(self, decoder_net, y_species_dim, y_embed_dim, y_amr_dim, likelihood='bernoulli', fixed_var=1.0, embedding=True):
        super().__init__()
        self.decoder = decoder_net
        self.y_embed = nn.Embedding(y_species_dim, y_embed_dim)
        self.y_amr_dim = y_amr_dim
        self.likelihood = likelihood
        self.var = fixed_var
        self.embedding = embedding
        self.input_dim = y_embed_dim if embedding else y_species_dim

    def get_cond(self, y_species, y_amr):
        return get_condition(y_species, self.input_dim, y_amr, self.y_embed, self.y_amr_dim, embedding=self.embedding)

    def decode(self, z, cond):
        """Low-level decoder call with precomputed cond."""
        return self.decoder(z, cond)

    def forward(self, z, y_species, y_amr=None):
        cond = self.get_cond(y_species, y_amr)
        logits = self.decode(z, cond)
        if self.likelihood == 'gaussian':
            return logits
        else:
            probs = torch.sigmoid(logits)
            return probs

    def sample(self, z, y_species, y_amr=None):
        cond = self.get_cond(y_species, y_amr)
        out = self.decode(z, cond)
        if self.likelihood == 'gaussian':
            std = torch.sqrt(torch.tensor(self.var)).to(out.device)
            eps = torch.randn_like(out)
            return out + eps * std
        else:
            return torch.sigmoid(out)

    def log_prob(self, x, z, cond):
        x_flat = x.view(x.size(0), -1)
        out = self.decode(z, cond)

        if self.likelihood == 'gaussian':
            log_prob = -0.5 * ((x_flat - out) ** 2) / self.var - 0.5 * torch.log(2 * torch.pi * torch.tensor(self.var))
            return log_prob.sum(dim=-1)
        else:
            probs = torch.sigmoid(out)
            log_prob = -F.binary_cross_entropy(probs, x_flat, reduction='none').sum(dim=1)
            return log_prob

class ConditionalPrior(nn.Module):
    def __init__(self, y_species_dim, y_embed_dim, y_amr_dim, latent_dim, prior_hidden1=128, prior_hidden2=64, embedding=True):
        super().__init__()
        self.y_embed = nn.Embedding(y_species_dim, y_embed_dim)
        self.y_amr_dim = y_amr_dim
        cond_dim = y_embed_dim + y_amr_dim if embedding else y_species_dim + y_amr_dim
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, prior_hidden1), nn.LeakyReLU(),
            nn.Linear(prior_hidden1, prior_hidden2), nn.LeakyReLU(),
            nn.Linear(prior_hidden2, 2 * latent_dim)
        )
        self.embedding = embedding
        self.input_dim = y_embed_dim if embedding else y_species_dim

    def get_cond(self, y_species, y_amr):
        return get_condition(y_species, self.input_dim, y_amr, self.y_embed, self.y_amr_dim, embedding=self.embedding)

    def forward(self, y_species, y_amr=None):
        cond = self.get_cond(y_species, y_amr)
        h = self.fc(cond)
        mu_p, log_var_p = torch.chunk(h, 2, dim=1)
        return mu_p, log_var_p

class ConditionalVAE(nn.Module):
    def __init__(self, encoder_net, decoder_net,
                y_species_dim, y_embed_dim, y_amr_dim,
                latent_dim, embedding=True, likelihood='bernoulli', fixed_var=1.0, beta=1.0):
        super().__init__()
        self.encoder = ConditionalEncoder(encoder_net, y_species_dim, y_embed_dim, y_amr_dim, embedding=embedding)
        self.decoder = ConditionalDecoder(decoder_net, y_species_dim, y_embed_dim, y_amr_dim, likelihood, fixed_var, embedding=embedding)
        self.prior = ConditionalPrior(y_species_dim, y_embed_dim, y_amr_dim, latent_dim, embedding=embedding)
        self.beta = beta

    def forward(self, x, y_species, y_amr=None):
        mu_e, log_var_e = self.encoder(x, y_species, y_amr)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        cond = self.decoder.get_cond(y_species, y_amr) # Precompute cond
        RE = self.decoder.log_prob(x, z, cond)
        mu_p, log_var_p = self.prior(y_species, y_amr)
        KL = -0.5 * torch.sum(1 + (log_var_e - log_var_p) - ((mu_e - mu_p).pow(2) + log_var_e.exp()) / log_var_p.exp(), dim=1)
        NLL = -(RE - self.beta * KL).mean()
        return NLL, KL.mean()
    
    def sample(self, y_species, y_amr=None):
        mu_p, log_var_p = self.prior(y_species, y_amr)
        std = torch.exp(0.5 * log_var_p)
        eps = torch.randn_like(std)
        z = mu_p + std * eps
        return self.decoder.sample(z, y_species, y_amr)
    
def generate_spectra_vae(model, n_samples, device=None, label_correspondence=None):
    """
    Generate n_samples spectra for each label using a ConditionalVAE.
    Args:
        model: Trained ConditionalVAE (must be in eval mode).
        n_samples: Number of spectra to generate per label.
        device: torch.device (optional, will use model's device if None).
        label_correspondence: dict mapping label indices to label names (or vice versa).
    Returns:
        dict: {label_name: tensor of generated spectra} if conditional, else tensor of generated spectra.
    """
    model.eval()
    device = device or next(model.parameters()).device

    if label_correspondence:
        results = {}
        for idx, label_name in label_correspondence.items():
            y_species = torch.full((n_samples,), idx, dtype=torch.long, device=device)
            generated = model.sample(y_species)
            results[label_name] = generated
            return results

    else:
        with torch.no_grad():
            # Sample from prior and decode
            z = model.prior.sample(n_samples).to(device)
            generated = model.decoder.sample(z)
        return generated
