"""
This is the main script to run experiments for training a GAN for synthetic MALDI-TOF spectra generation.

Training is done using datasets MARISMa (2018-2023) and DRIAMS (A, B).
Evaluation can include generating spectra and testing with MARISMa 2024.

"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.data import load_data, get_dataloaders
from utils.training_utils import setuplogging

# Data
pickle_driams = "pickles/DRIAMS_study.pkl"
pickle_marisma = "pickles/MARISMa_study.pkl"
name = "gan_pretrainedGenerator"
results_path = os.path.join("results", name)
os.makedirs(results_path, exist_ok=True)
plot_path = os.path.join(results_path, "plots")
os.makedirs(plot_path, exist_ok=True)
mode = 'training'  # or 'evaluation'
batch_size = 128
logger = setuplogging(name, mode, results_path)

train, val, test, ood = load_data(pickle_marisma, pickle_driams, logger)
train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size)

# GAN
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
latent_dim = 32
image_dim = 6000
label_dim = 10
epochs = 30
num_layers = 3
output_dim = image_dim

# Path to VAE checkpoint (contains 'encoder', 'decoder', 'prior')
vae_ckpt_path = '/Volumes/usuarios_ml4ds/lschmidt/GITHUB/MALDIVAS/results/vae_MLP3_32/best_model_vae_MLP3_32.pt'


# Generation Network (p_theta)
class MLPDecoder1D_Generator(nn.Module):
    def __init__(self, latent_dim, num_layers, output_dim, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.in_dim = latent_dim + cond_dim

        layers = []
        for i in range(num_layers):
            out_dim = 2 ** (i + 2) * latent_dim  # e.g., 4×, 8×, 16×...
            layers.append(nn.Linear(self.in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            self.in_dim = out_dim

        layers.append(nn.Linear(self.in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, cond=None):
        if cond is not None:
            z = torch.cat([z, cond], dim=1)  # shape: [batch, latent_dim + cond_dim]
        return self.net(z)

# Discriminator: D(x, y)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(image_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        h = F.leaky_relu(self.fc3(h), 0.2)
        return torch.sigmoid(self.fc_out(h))



# Instantiate generator and discriminator before loading weights
generator = MLPDecoder1D_Generator(latent_dim, num_layers, output_dim).to(device)
discriminator = Discriminator().to(device)

# --- Load pretrained decoder weights from VAE checkpoint into generator ---
vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
# Reconstruct decoder state dict from flat keys if needed
decoder_prefix = 'decoder.decoder.'
decoder_state_dict = {k[len(decoder_prefix):]: v for k, v in vae_ckpt.items() if k.startswith(decoder_prefix)}
if decoder_state_dict:
    generator.load_state_dict(decoder_state_dict)
    print('Loaded pretrained decoder weights from VAE checkpoint into generator (from flat keys, prefix stripped).')
else:
    raise KeyError(f"No keys starting with '{decoder_prefix}' found in checkpoint: {vae_ckpt_path}")

# Loss and optimizers
criterion = nn.BCELoss()
lr_g = 2e-4
lr_d = 1e-4
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

# --- Freeze generator and pretrain discriminator ---
for param in generator.parameters():
    param.requires_grad = False

pretrain_epochs = 5  # number of epochs to pretrain discriminator
for epoch in range(pretrain_epochs):
    epoch_d_loss = 0.0
    num_batches = 0
    for x_real in train_loader:
        x_real = x_real.to(device)
        batch_size = x_real.size(0)
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        z = torch.randn(batch_size, latent_dim).to(device)
        x_fake = generator(z).detach()
        d_real = discriminator(x_real)
        d_fake = discriminator(x_fake)
        d_loss = criterion(d_real, valid) + criterion(d_fake, fake)
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        epoch_d_loss += d_loss.item()
        num_batches += 1
    print(f"[Pretrain] Epoch {epoch+1}: D Loss = {epoch_d_loss / num_batches:.4f}")
    logger.info(f"[Pretrain] Epoch {epoch+1}: D Loss = {epoch_d_loss / num_batches:.4f}")

# --- Unfreeze generator and continue GAN training ---
for param in generator.parameters():
    param.requires_grad = True

loss_history = {
    "D": [],
    "G": []
}
for epoch in range(pretrain_epochs, epochs):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    num_batches = 0
    for x_real in train_loader:
        x_real = x_real.to(device)
        batch_size = x_real.size(0)
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        x_fake = generator(z).detach()
        d_real = discriminator(x_real)
        d_fake = discriminator(x_fake)
        d_loss = criterion(d_real, valid) + criterion(d_fake, fake)
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        x_gen = generator(z)
        d_gen = discriminator(x_gen)
        g_loss = criterion(d_gen, valid)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        num_batches += 1

    loss_history["D"].append(epoch_d_loss / num_batches)
    loss_history["G"].append(epoch_g_loss / num_batches)
    print(f"Epoch {epoch+1}: D Loss = {loss_history['D'][-1]:.4f} | G Loss = {loss_history['G'][-1]:.4f}")
    logger.info(f"Epoch {epoch+1}: D Loss = {loss_history['D'][-1]:.4f} | G Loss = {loss_history['G'][-1]:.4f}")

# Save model state_dicts
torch.save(generator.state_dict(), os.path.join(results_path, "generator_state_dict.pt"))
torch.save(discriminator.state_dict(), os.path.join(results_path, "discriminator_state_dict.pt"))

# Plot training curve (D and G losses)
epochs_range = range(1, len(loss_history['D']) + 1)
plt.figure()
plt.plot(epochs_range, loss_history['D'], label='D loss')
plt.plot(epochs_range, loss_history['G'], label='G loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'training_curve.png'))
plt.close()

# Generate and save sample images
n_samples = 6
z = torch.randn(n_samples, latent_dim).to(device)
with torch.no_grad():
    samples = generator(z).cpu()  # [N, image_dim], already passed through sigmoid in generator


# Plot generated spectra as line plots in a grid
nrow = 8
ncol = int(np.ceil(n_samples / nrow))
fig, axes = plt.subplots(ncol, nrow, figsize=(2.5 * nrow, 2 * ncol), sharex=True, sharey=True)
axes = axes.flatten()
for i in range(n_samples):
    axes[i].plot(samples[i].numpy(), color='black', linewidth=1)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(f"Sample {i+1}", fontsize=8)
for i in range(n_samples, len(axes)):
    axes[i].axis('off')
plt.suptitle('Generated Spectra', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(plot_path, 'generated_spectra.png'))
plt.close()

print(f"Saved models and plots to {results_path}")
