import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

from PIKE import PIKE, generate_spectrum, reshape_spectrum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.probability_distributions import log_bernoulli, log_normal_diag

def loss_function(recon_x, x, mu_q, logvar_q, loss_mode, reduction='sum'):
    # Calculate KL divergence loss.
    kl = kl_divergence(mu_q, logvar_q, reduction=reduction)
    
    # Calculate reconstruction error.
    rec = rec_loss(recon_x, x, loss_mode, reduction=reduction)
    
    return rec + kl, rec, kl

def kl_divergence(mu_q, log_var_q, reduction='sum'):
    """
    Compute KL divergence KL(q(z|x) || p(z)) between two distributions:
    q(z|x) = N(mu_q, std_q)
    p(z) = N(mu_p, std_p)

    KL(q(z|x) || p(z)) = 0.5 * (logvar_p - logvar_q + (std_q^2 + (mu_q - mu_p)^2) / std_p^2 - 1)
    where mu_p = 0, std_p = 1

    Args:
        mu_q: Mean of the posterior distribution q(z|x)
        log_var_q: Log variance of the posterior distribution q(z|x)
        reduction: Reduction method ('avg' or 'sum')
    Returns:
        KL divergence value
    """

    # Define mu and std of the prior distribution.
    mu_p = torch.zeros_like(mu_q)
    logvar_p = torch.zeros_like(log_var_q)

    # Get the std
    std_q = torch.exp(0.5 * log_var_q)
    std_p = torch.exp(0.5 * logvar_p)

    # Create the posterior and prior distributions
    q = torch.distributions.Normal(mu_q, std_q)
    p = torch.distributions.Normal(mu_p, std_p)

    # Calculate KL(q||p)
    kl = torch.distributions.kl_divergence(q, p)  # shape [batch, dim]

    if reduction == 'avg':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl

def rec_loss(recon_x, x, loss_mode, reduction='sum'):

    # Choose the reconstruction loss based on the keyword.
    if loss_mode == 'bce':
        RE = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    elif loss_mode == 'mse':
        RE = F.mse_loss(recon_x, x, reduction=reduction)
    elif loss_mode == 'gaussian':
        # Assuming unit variance, the negative log-likelihood is proportional to MSE.
        RE = 0.5 * F.mse_loss(recon_x, x, reduction=reduction)
    elif loss_mode == 'pike':
        i_real, mz_real = x
        i_synth, mz_synth = recon_x
        batch_size = i_real.shape[0]
        pike = PIKE(t=8)
        losses = []
        for b in range(batch_size):
            # Convert to numpy arrays
            mz_real_np = mz_real[b].detach().cpu().numpy()
            i_real_np = i_real[b].detach().cpu().numpy()
            mz_synth_np = mz_synth[b].detach().cpu().numpy()
            i_synth_np = i_synth[b].detach().cpu().numpy()
            X_mz, X_i = reshape_spectrum(mz_real_np, i_real_np)
            Y_mz, Y_i = reshape_spectrum(mz_synth_np, i_synth_np)
            _, _, K = pike(X_mz, X_i, Y_mz, Y_i)
            _, _, K_real = pike(X_mz, X_i)
            _, _, K_fake = pike(Y_mz, Y_i)
            K_norm = K / np.sqrt(K_real * K_fake)
            print(f"Normalized PIKE loss for batch {b}: {K_norm[0][0]}")
            print(f"PIKE loss for batch {b}: {1 - K_norm[0][0]}")
            losses.append(1 - K_norm[0][0])
        RE = torch.tensor(losses, device=i_real.device).mean()
        print(f"Average PIKE loss: {RE.item()}")
    else:
        raise ValueError("Unsupported loss mode. Use 'bce', 'mse' or 'gaussian'.")
    
    return RE
