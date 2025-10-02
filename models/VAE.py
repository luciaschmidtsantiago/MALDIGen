import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.conditional_utils import get_condition, impute_missing_labels, compute_attr_prediction_loss


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
        theta = self.decode(z)  # Use sigmoid output
        print(f"x min: {x.min()}, x max: {x.max()}")
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
    


class ConditionalEncoder(nn.Module):
    def __init__(self, encoder_net, y_dim, y_embed_dim, label2_dim):
        super().__init__()
        self.encoder = encoder_net
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.label2_dim = label2_dim
    
    def forward(self, x, cond):
        h_e = self.encoder(x, cond)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e
    
    def sample(self, x=None, cond=None, mu_e=None, log_var_e=None):
        if (mu_e is None) or (log_var_e is None):
            mu_e, log_var_e = self.forward(x, cond)
        std = torch.exp(0.5 * log_var_e)
        eps = torch.randn_like(std)
        return mu_e + std * eps

class ConditionalDecoder(nn.Module):
    def __init__(self, decoder_net, y_dim, y_embed_dim, label2_dim, likelihood='bernoulli', fixed_var=1.0):
        super().__init__()
        self.decoder = decoder_net
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.label2_dim = label2_dim
        self.likelihood = likelihood
        self.var = fixed_var
    
    def get_cond(self, y, label2):
        """Método wrapper que usa la función común get_condition"""
        return get_condition(y, label2, self.y_embed, self.label2_dim)
    
    def forward(self, z, cond):
        out = self.decoder(z, cond)
        if self.likelihood == 'gaussian':
            mu = out
            return mu
        else:
            logits = out
            probs = torch.sigmoid(logits)
            return probs
    
    def sample(self, z, cond):
        if self.likelihood == 'gaussian':
            mu = self.forward(z, cond)
            std = torch.sqrt(torch.tensor(self.var)).to(mu.device)
            eps = torch.randn_like(mu)
            return mu + eps * std
        else:
            probs = self.forward(z, cond)
            return probs
    
    def log_prob(self, x, z, cond):
        x_flat = x.view(x.size(0), -1)
        if self.likelihood == 'gaussian':
            mu = self.forward(z, cond)
            log_prob = -0.5 * ((x_flat - mu) ** 2) / self.var - 0.5 * torch.log(2 * torch.pi * torch.tensor(self.var))
            return log_prob.sum(dim=-1)
        else:
            theta = self.forward(z, cond)
            log_prob = -F.binary_cross_entropy(theta, x_flat, reduction='none').sum(dim=1)
            return log_prob

class ConditionalPrior(nn.Module):
    def __init__(self, y_dim, y_embed_dim, label2_dim, latent_dim, prior_hidden1=128, prior_hidden2=64):
        super().__init__()
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.label2_dim = label2_dim
        self.fc = nn.Sequential(
            nn.Linear(y_embed_dim + label2_dim, prior_hidden1), nn.LeakyReLU(),
            nn.Linear(prior_hidden1, prior_hidden2), nn.LeakyReLU(),
            nn.Linear(prior_hidden2, 2 * latent_dim)
        )
    
    def forward(self, cond):
        h = self.fc(cond)
        mu_p, log_var_p = torch.chunk(h, 2, dim=1)
        return mu_p, log_var_p

class ConditionalVAE(nn.Module):
    def __init__(self, encoder, decoder, prior, beta=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta = beta
    def forward(self, x, y, label2):
        cond = get_condition(y, label2, self.encoder.y_embed, self.encoder.label2_dim)
        mu_e, log_var_e = self.encoder(x, cond)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        RE = self.decoder.log_prob(x, z, cond)
        mu_p, log_var_p = self.prior(cond)
        KL = -0.5 * torch.sum(1 + (log_var_e - log_var_p) - ((mu_e - mu_p).pow(2) + log_var_e.exp()) / log_var_p.exp(), dim=1)
        vae_loss = -(RE - self.beta * KL)
        return vae_loss.mean()
    def sample(self, y, label2, batch_size=64):
        cond = get_condition(y, label2, self.encoder.y_embed, self.encoder.label2_dim)
        mu_p, log_var_p = self.prior(cond)
        std = torch.exp(0.5 * log_var_p)
        eps = torch.randn_like(std)
        z = mu_p + std * eps
        return self.decoder.sample(z, cond)
    

class SemisupervisedConditionalVAE(ConditionalVAE):
    def __init__(self, encoder, decoder, prior, attr_predictor, alpha=1.0, beta=1.0, missing_strategy='soft'):
        # Inicializar ConditionalVAE base
        super().__init__(encoder, decoder, prior, beta)
        
        # Inferir parámetros automáticamente desde los componentes
        y_dim = encoder.y_embed.num_embeddings
        y_embed_dim = encoder.y_embed.embedding_dim
        self.label2_dim = encoder.label2_dim
        
        # Añadir componentes específicos para semisupervised
        self.attr_predictor = attr_predictor
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.alpha = alpha         # peso para attr_prediction loss
        
        # missing_strategy controla cómo manejar missing values para el VAE:
        # - 'soft': Usar predicciones como probabilidades soft → embedding ponderado
        # - 'hard': Umbralizar predicciones a 0/1 → embedding tradicional  
        # - 'ignore': Missing → 0, solo usar observados → más conservador
        assert missing_strategy in ['soft', 'hard', 'ignore'], f"missing_strategy debe ser 'soft', 'hard' o 'ignore', got {missing_strategy}"
        self.missing_strategy = missing_strategy
    
    def forward(self, x, y, label2):
        # 1. Imputar missing values según la estrategia configurada
        y_input = impute_missing_labels(x, y, label2, self.attr_predictor, self.y_embed, self.label2_dim, self.missing_strategy)
        
        # 2. El VAE usa directamente las etiquetas procesadas
        # get_condition automáticamente detecta si son soft o hard labels
        vae_loss = super().forward(x, y_input, label2)
        
        # 3. Pérdida de predicción siempre sobre atributos observados reales
        attr_pred_loss = compute_attr_prediction_loss(x, y, label2, self.attr_predictor, self.y_embed, self.label2_dim)
        
        total_loss = vae_loss + self.alpha * attr_pred_loss
        return total_loss
    
    def sample(self, y, label2, batch_size=64, fill_missing='neutral'):
        """
        Generar muestras reutilizando ConditionalVAE.sample después de imputar
        fill_missing: 'neutral' (missing → 0.5) o 'predictor' (usar attr_predictor)
        """
        # Imputar missing values para generación
        y_filled = y.clone().float()
        
        for i in range(y.size(1)):
            missing_idx = (y[:, i] == -1).nonzero(as_tuple=True)[0]
            if len(missing_idx) > 0:
                if fill_missing == 'predictor':
                    # Usar predictor (necesita input x dummy para generación)
                    x_dummy = torch.zeros((len(missing_idx), 9701), device=y.device)  # TODO: hacer dinámico
                    y_idx = torch.full((len(missing_idx),), i, dtype=torch.long, device=y.device)
                    y_emb = self.y_embed(y_idx)
                    label2_missing = label2[missing_idx]
                    label2_onehot = F.one_hot(label2_missing, num_classes=self.label2_dim).float()
                    attr_input = torch.cat([x_dummy, y_emb, label2_onehot], dim=1)
                    pred = self.attr_predictor(attr_input)
                    y_filled[missing_idx, i] = pred.squeeze(-1)
                else:
                    y_filled[missing_idx, i] = 0.5  # neutral/incierto
        
        # Reutilizar el método sample del ConditionalVAE padre
        return super().sample(y_filled, label2, batch_size)
    