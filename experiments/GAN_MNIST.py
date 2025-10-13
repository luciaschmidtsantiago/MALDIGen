"""
This is the main script to run experiments for training a GAN for synthetic MALDI-TOF spectra generation.

Training is done using datasets MARISMa (2018-2023) and DRIAMS (A, B).
Evaluation can include generating spectra and testing with MARISMa 2024.

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten from 28x28 => 784
])
full_train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# Split into train and validation sets (90% train, 10% validation)
valid_size = 0.1  # 10% for validation
num_train = len(full_train_data)
split = int(np.floor(valid_size * num_train))
train, val = random_split(full_train_data, [num_train - split, split])

# Test data
test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# DataLoader for training, validation, and test sets.
train_loader = DataLoader(train, batch_size=128, shuffle=True)
val_loader = DataLoader(val, batch_size=128, shuffle=False)
test_loader = DataLoader(test, batch_size=128, shuffle=False)



# GAN
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
latent_dim = 20
image_dim = 28 * 28
label_dim = 10
batch_size = 128
epochs = 10
lr = 1e-3

# Generation Network (p_theta)
class GenerationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc_out = nn.Linear(1024, image_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return torch.sigmoid(self.fc_out(h))


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

# Initialize models
generator = GenerationNetwork().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

loss_history = {
    "D": [],
    "G": []
}

for epoch in range(epochs):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    num_batches = 0

    for x_real, labels in train_loader:
        x_real = x_real.to(device)
        labels = labels.to(device)
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

        # Accumulate losses
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        num_batches += 1

    # Save average losses
    loss_history["D"].append(epoch_d_loss / num_batches)
    loss_history["G"].append(epoch_g_loss / num_batches)

    print(f"Epoch {epoch+1}: D Loss = {loss_history['D'][-1]:.4f} | G Loss = {loss_history['G'][-1]:.4f}")

# Create results folder and save models + plots
results_dir = os.path.join("results", "gan_mnist")
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Save model state_dicts
torch.save(generator.state_dict(), os.path.join(results_dir, "generator_state_dict.pt"))
torch.save(discriminator.state_dict(), os.path.join(results_dir, "discriminator_state_dict.pt"))

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
plt.savefig(os.path.join(plots_dir, 'training_curve.png'))
plt.close()

# Generate and save sample images
n_samples = 64
z = torch.randn(n_samples, latent_dim).to(device)
with torch.no_grad():
    samples = generator(z).cpu()  # [N, image_dim], already passed through sigmoid in generator

# reshape to images [N,1,28,28]
images = samples.view(-1, 1, 28, 28)
grid = vutils.make_grid(images, nrow=8, normalize=True, pad_value=1)
plt.figure(figsize=(8, 8))
# grid is C x H x W
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
plt.axis('off')
plt.title('Generated samples')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'generated_samples.png'))
plt.close()

print(f"Saved models and plots to {results_dir}")
