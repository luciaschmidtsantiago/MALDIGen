# ConditionalEncoder, ConditionalDecoder, ConditionalPrior, ConditionalVAE
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_condition(y, label2, y_embed_layer, label2_dim):
    """
    Método común para generar condiciones a partir de etiquetas multilabel y categóricas.
    Ahora soporta tanto etiquetas binarias (0/1) como probabilidades soft [0,1].
    
    Args:
        y: tensor [batch, num_labels] con etiquetas multilabel 
           - Valores 0/1: etiquetas binarias (comportamiento original)
           - Valores [0,1]: probabilidades soft (nuevo comportamiento)
        label2: tensor [batch] con etiquetas categóricas
        y_embed_layer: nn.Embedding layer para las etiquetas multilabel
        label2_dim: número de categorías para label2
    
    Returns:
        cond: tensor [batch, y_embed_dim + label2_dim] con la condición concatenada
    """
    # Procesar etiquetas multilabel (con soporte para soft labels)
    batch_embeds = []
    
    # Obtener todos los embeddings de una vez para eficiencia
    all_label_embeds = y_embed_layer.weight  # [num_labels, embed_dim]
    
    for i in range(y.size(0)):
        y_sample = y[i]  # [num_labels]
        
        # Si hay probabilidades soft (valores entre 0 y 1), usar ponderación
        # Si son binarias (solo 0s y 1s), comportamiento original
        if torch.any((y_sample > 0) & (y_sample < 1)):
            # SOFT LABELS: Ponderar embeddings por probabilidades
            # y_sample[j] * all_label_embeds[j] para cada label j
            weighted_embeds = y_sample.unsqueeze(1) * all_label_embeds  # [num_labels, embed_dim]
            
            # Suma ponderada normalizada por la suma de probabilidades
            prob_sum = y_sample.sum()
            if prob_sum > 0:
                y_emb_sample = weighted_embeds.sum(dim=0) / prob_sum  # [embed_dim]
            else:
                y_emb_sample = torch.zeros(y_embed_layer.embedding_dim, device=y.device)
        else:
            # HARD LABELS: Comportamiento original (solo 0s y 1s)
            idxs = (y_sample == 1).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                y_emb_sample = torch.zeros(y_embed_layer.embedding_dim, device=y.device)
            else:
                embeds = y_embed_layer(idxs)
                y_emb_sample = embeds.mean(dim=0)
        
        batch_embeds.append(y_emb_sample)
    
    y_emb = torch.stack(batch_embeds, dim=0)
    
    # Procesar etiquetas categóricas
    label2_onehot = F.one_hot(label2, num_classes=label2_dim).float()
    
    # Concatenar ambas representaciones
    cond = torch.cat([y_emb, label2_onehot], dim=1)
    return cond

class ConditionalEncoder(nn.Module):
    def __init__(self, encoder_net, y_dim, y_embed_dim, label2_dim):
        super().__init__()
        self.encoder = encoder_net
        self.y_embed = nn.Embedding(y_dim, y_embed_dim)
        self.label2_dim = label2_dim
    
    def get_cond(self, y, label2):
        """Método wrapper que usa la función común get_condition"""
        return get_condition(y, label2, self.y_embed, self.label2_dim)
    
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
    
    def get_cond(self, y, label2):
        """Método wrapper que usa la función común get_condition"""
        return get_condition(y, label2, self.y_embed, self.label2_dim)
    
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
        cond = self.encoder.get_cond(y, label2)
        mu_e, log_var_e = self.encoder(x, cond)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        RE = self.decoder.log_prob(x, z, cond)
        mu_p, log_var_p = self.prior(cond)
        KL = -0.5 * torch.sum(1 + (log_var_e - log_var_p) - ((mu_e - mu_p).pow(2) + log_var_e.exp()) / log_var_p.exp(), dim=1)
        vae_loss = -(RE - self.beta * KL)
        return vae_loss.mean()
    def sample(self, y, label2, batch_size=64):
        cond = self.encoder.get_cond(y, label2)
        mu_p, log_var_p = self.prior(cond)
        std = torch.exp(0.5 * log_var_p)
        eps = torch.randn_like(std)
        z = mu_p + std * eps
        return self.decoder.sample(z, cond)
    