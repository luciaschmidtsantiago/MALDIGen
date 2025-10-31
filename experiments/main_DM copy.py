import os
import sys
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VANESSA.diffusion_utilities import *
from dataloader.data import load_data, get_dataloaders, compute_mean_spectra_per_label
from utils.visualization import plot_generated_spectra_per_label, plot_meanVSgenerated
from losses.PIKE_GPU import calculate_pike_matrix, calculate_PIKE_gpu

class ContextUnet1D(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, length=256):
        super(ContextUnet1D, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.length = length

        # Initial 1D convolutional block
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU()
        )

        # Down-sampling path
        self.down1 = nn.Sequential(
            nn.Conv1d(n_feat, n_feat, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(n_feat, 2 * n_feat, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )

        # Bottleneck
        self.to_vec = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.GELU()
        )

        # Embeddings
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, n_feat)

        # Up-sampling path
        self.up0 = nn.Sequential(
            nn.ConvTranspose1d(2 * n_feat, 2 * n_feat, kernel_size=length // 4, stride=length // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(4 * n_feat, n_feat, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(2 * n_feat, n_feat, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU()
        )

        # Output layer
        self.out = nn.Sequential(
            nn.Conv1d(2 * n_feat, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv1d(n_feat, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, c=None):
        # x: (batch, in_channels, length)
        # t: (batch, 1)
        # c: (batch, n_cfeat)
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat, device=x.device, dtype=x.dtype)

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(torch.cat([cemb1 * up1 + temb1, down2], dim=1))
        up3 = self.up2(torch.cat([cemb2 * up2 + temb2, down1], dim=1))
        out = self.out(torch.cat([up3, x], dim=1))
        return out

def perturb_input(x, t, noise, ab_t):
    """
    Perturbs a real image x at timestep t using the DDPM noise schedule ab_t.
    Supports x of shape (B, C, H, W) or (B, C, L) where L=H*W.
    - x: torch.Tensor, shape (B, C, H, W) or (B, C, L)
    - t: torch.Tensor, shape (B,) or int
    - noise: torch.Tensor, same shape as x
    - ab_t: torch.Tensor, cumulative product of alphas, shape (timesteps+1,)
    Returns: perturbed image (torch.Tensor, same shape as x)
    """
    # Get batch size
    B = x.shape[0]
    # Prepare ab for broadcasting
    if isinstance(t, torch.Tensor):
        ab = ab_t[t].view(B, 1, 1, 1) if x.ndim == 4 else ab_t[t].view(B, 1, 1)
    else:
        ab = ab_t[t].view(1, 1, 1, 1) if x.ndim == 4 else ab_t[t].view(1, 1, 1)
    return x * ab.sqrt() + noise * (1 - ab).sqrt()

def denormalize_spectra(x):
    """
    Convert spectra from [-1, 1] back to [0, 1].
    Works on torch tensors or numpy arrays.
    """
    if isinstance(x, torch.Tensor):
        return (x + 1.0) / 2.0
    elif isinstance(x, np.ndarray):
        return (x + 1.0) / 2.0
    else:
        raise TypeError(f"Unsupported type {type(x)} for denormalization")

def generate_spectra_per_label_ddpm(
    model, label_correspondence, n_samples, timesteps, a_t, b_t, ab_t, device
):
    """
    Generate n_samples per label using the trained diffusion model.
    """
    model.eval()
    results = {}
    num_classes = len(label_correspondence)

    for label_id, label_name in label_correspondence.items():
        print(f"Generating diffusion samples for label: {label_name}")

        # --- Create correct one-hot context for that label ---
        c = torch.zeros(n_samples, num_classes, device=device)
        c[:, label_id] = 1.0

        # --- Start from Gaussian noise ---
        L = model.length
        x = torch.randn(n_samples, model.in_channels, L, device=device)

        # --- Diffusion sampling ---
        with torch.no_grad():
            for t_inv in range(timesteps, 0, -1):
                t = torch.full((n_samples,), t_inv, device=device, dtype=torch.long)
                t_norm = (t.float() / float(timesteps)).view(-1, 1)
                eps = model(x, t_norm, c)
                ab = ab_t[t].view(n_samples, 1, 1)
                a = a_t[t].view(n_samples, 1, 1)
                b = b_t[t].view(n_samples, 1, 1)
                x = (x - (b / (1 - ab).sqrt()) * eps) / a.sqrt()
                if t_inv > 1:
                    x += b.sqrt() * torch.randn_like(x)

        results[label_name] = x.detach().cpu()

    return results

# Define global label names and colors (if not already defined)
LABEL_NAMES = [
    'Enterobacter_cloacae_complex',
    'Enterococcus_Faecium',
    'Escherichia_Coli',
    'Klebsiella_Pneumoniae',
    'Pseudomonas_Aeruginosa',
    'Staphylococcus_Aureus',
]
LABEL_TO_COLOR = {lbl: plt.cm.Spectral(i / 5) for i, lbl in enumerate(LABEL_NAMES)}

def plot_generated_vs_all_means(generated_sample, mean_spectra_dict, label_correspondence, save_path):
    """
    Plot a single generated spectrum vs. the mean spectra of all labels (6 subplots).
    Each subplot: black dashed = generated, colored line = class mean.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ensure shapes
    if generated_sample.ndim == 3:
        generated_sample = generated_sample.squeeze(0).squeeze(0)
    elif generated_sample.ndim == 2:
        generated_sample = generated_sample.squeeze(0)
    generated_sample = generated_sample.to(device).float()

    # Order labels by their defined sequence (LABEL_NAMES)
    ordered_items = sorted(mean_spectra_dict.items(), key=lambda kv: LABEL_NAMES.index(label_correspondence[kv[0]]))

    fig, axes = plt.subplots(len(LABEL_NAMES), 1, figsize=(10, 2.5 * len(LABEL_NAMES)), sharex=True)

    if len(LABEL_NAMES) == 1:
        axes = [axes]

    for i, (label_id, mean_spec) in enumerate(ordered_items):
        label_name = label_correspondence[label_id]
        color = LABEL_TO_COLOR.get(label_name, 'C0')
        ax = axes[i]

        mean_spec = mean_spec.squeeze().to(device).float()

        # Compute PIKE distance between mean and generated spectrum
        try:
            pike_val = calculate_PIKE_gpu(mean_spec, generated_sample)
        except Exception:
            pike_val = float('nan')

        # Plot mean (colored) and generated (gray dashed)
        ax.plot(generated_sample.cpu().numpy(), color='lightgray', linestyle='--', linewidth=1.2, label='Generated')
        ax.plot(mean_spec.cpu().numpy(), color=color, linewidth=2.0, label=f'Mean {label_name}', alpha=0.7)

        ax.set_ylabel("Intensity", fontsize=9)
        ax.set_title(f"{label_name}  (PIKE={pike_val:.4f})", fontsize=10)
        ax.legend(loc='upper right', fontsize='x-small')

    axes[-1].set_xlabel("m/z index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved combined comparison plot → {save_path}")

##### DATA LOADING ######
pickle_driams = "pickles/DRIAMS_study.pkl"
pickle_marisma = "pickles/MARISMa_study.pkl"

# Default batch size for dataloaders (can be overridden below in hyperparameters)
batch_size = 64

train, val, test, ood = load_data(pickle_marisma, pickle_driams, get_labels=True, model_type='diffusion')
train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size=batch_size)

def _extract_x(batch):
    # dataloader may return x or (x, y) tuples
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

##### DATA INSPECTION ######
N_batches = 50  # limit how many batches we scan (faster)
vals = []
count = 0
for b in train_loader:
    x = _extract_x(b)
    x_cpu = x.detach().cpu().view(x.size(0), -1).numpy()
    vals.append(x_cpu)
    count += 1
    if count >= N_batches:
        break

if len(vals) == 0:
    print('Warning: train_loader yielded no batches')
else:
    arr = np.concatenate(vals, axis=0).ravel()
    print('Data samples scanned:', arr.size)
    print(f'min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6f}')
    # show quantiles to detect zeros / saturated ranges
    q = np.quantile(arr, [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0])
    print('quantiles [0,0.1,0.5,0.9,1]:', q[[0,3,4,5,8]])

# Save a small example for visual inspection later
example_x = None
for b in train_loader:
    x = _extract_x(b)
    example_x = x[0].detach().cpu()
    break
print('Example sample shape:', None if example_x is None else tuple(example_x.shape))

# Diagnostic: labels, shape, ranges
for i, batch in enumerate(train_loader):
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        print(f"Batch {i}: x shape={x.shape}, y shape={y.shape}")
        print(f"  x min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
        print(f"  y min={y.min().item():.4f}, max={y.max().item():.4f}, dtype={y.dtype}")
        print(f"  Example label: {y[0]}")
    else:
        x = batch
        print(f"Batch {i}: x shape={x.shape}")
        print(f"  x min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
    if i >= 2:
        break  # Solo muestra los primeros lotes para no saturar la salida

print("Dimensions of x:", x.shape)
if 'y' in locals():
    try:
        print("Dimensions of y:", y.shape)
    except Exception:
        print("Labels present but unable to print their shape.")
else:
    print("No labels found in loader (y undefined).")


###### DEVICE SETUP ######
print(f"PyTorch version: {torch.__version__}")

# Verificar CUDA (NVIDIA GPU)
if torch.cuda.is_available():
    print("CUDA available")
    print(f"   - Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"     Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    print(f"   - Current device: {torch.cuda.current_device()}")
    device = "cuda:0"
else:
    print("CUDA not available")
    device = "cpu"

print(f"📱 Device seleccionado: {device}")


##### HYPERPARAMETERS SETUP ######
# Diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02
n_channels = 1 # grayscale images

# Network hyperparameters
n_feat = 64 # 64 hidden dimension feature
n_cfeat = y.shape[1] # context vector is of size 5
height = 16 # 16x16 image
results_path = os.path.join('results', 'dm', 'DM_training')
os.makedirs(results_path, exist_ok=True)

# Training hyperparameters
n_epoch = 32
lrate=1e-3

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device, dtype=torch.float32) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1.0


##### MODEL SETUP ######
nn_model = ContextUnet1D(in_channels=n_channels, n_feat=n_feat, n_cfeat=n_cfeat, length=x.shape[2]).to(device)
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

# Model summary (optional)

try:
    from pytorch_model_summary import summary
    print(
        "\n" + summary(
            nn_model,
            torch.zeros((1, n_channels, x.shape[2]), device=device),
            torch.zeros((1, 1), device=device),
            torch.zeros((1, n_cfeat), device=device),
            show_input=True
        )
    )
except Exception as e:
    print(f"Could not print model summary: {e}")
print(f"Total parameters: {sum(p.numel() for p in nn_model.parameters())/1e6:.2f}M")

##### TRAININING LOOP ######
training = True 
if training:
    nn_model.train()

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        # accumulate per-iteration losses for this epoch
        epoch_iter_losses = []

        #pbar = tqdm(dataloader, mininterval=2 )
        pbar = tqdm(train_loader, mininterval=2 )
        
        for x, c in pbar:   # x: images  c: context
            optim.zero_grad()
            x = x.to(device)
            # ensure context is float32 on the correct device (fix MPS linear dtype error)
            if c is None:
                c = torch.zeros(x.shape[0], n_cfeat, dtype=torch.float32, device=device)
            else:
                c = c.float().to(device)

            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0], dtype=torch.float32, device=device) + 0.9)
            c = c * context_mask.unsqueeze(-1)

            # perturb data
            noise = torch.randn_like(x, dtype=torch.float32, device=x.device)
            t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device, dtype=torch.long)
            x_pert = perturb_input(x, t, noise, ab_t)

            # also prepare normalized float timestep for the model, on correct device
            t_norm_for_model = (t.float() / float(timesteps)).view(-1, 1).float().to(device)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t_norm_for_model, c=c)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

            # record this iteration's loss so we can compute an epoch MSE
            try:
                epoch_iter_losses.append(float(loss.detach().cpu().item()))
            except Exception:
                epoch_iter_losses.append(loss.item())

        # compute and print epoch MSE (mean of per-iteration MSEs)
        if len(epoch_iter_losses) > 0:
            epoch_mse = sum(epoch_iter_losses) / len(epoch_iter_losses)
        else:
            epoch_mse = float('nan')
        print(f'--> Epoch {ep} MSE: {epoch_mse:.6e}')

        # save model periodically
        if ep%4==0 or ep == int(n_epoch-1):
            torch.save(nn_model.state_dict(), results_path + f"context_model_{ep}.pth")
            print('saved model at ' + results_path + f"context_model_{ep}.pth")

else: 
    ###### MODEL LOADING ######
    model_path = 'VANESSA/context_model_28.pth'
    print("Model path:", model_path)
    print("Exists:", os.path.exists(model_path))
    print("Is file:", os.path.isfile(model_path))
    print("Size (bytes):", os.path.getsize(model_path) if os.path.exists(model_path) else "N/A")

    nn_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model weights from {model_path}")


##### SAMPLE GENERATION ######
generation = True
plots_path = os.path.join(results_path, 'plots')
if generation:

    label_correspondence = train.label_convergence
    label_ids = sorted(label_correspondence.keys())
    label_names = [label_correspondence[i] for i in label_ids]
    print(f"Generating conditional samples for {len(label_names)} labels: {label_names}")

    # --- Compute mean spectra from train set ---
    train_loader_idx = []
    for x, y in train_loader:
        if y.ndim > 1:
            y = y.argmax(dim=1)
        train_loader_idx.append((x, y))
    # Then pass that temporary loader
    mean_std_spectra = compute_mean_spectra_per_label(train_loader_idx, device)


    # --- Denormalize mean spectra from [-1,1] → [0,1] and fix shape to [1, 6000]
    for k in mean_spectra_train:
        arr = denormalize_spectra(mean_spectra_train[k])
        # arr: [1, 1, 6000] -> [1, 6000]
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        mean_spectra_train[k] = arr

    mean_spectra_list = [mean_spectra_train[i] for i in range(len(mean_spectra_train))]


    # --- Generate spectra per label (Diffusion) ---
    n_generate = 500  # For PIKE matrix
    start_all = time.time()
    generated_spectra = generate_spectra_per_label_ddpm(nn_model, label_correspondence, n_generate, timesteps, a_t, b_t, ab_t, device)
    end_all = time.time()
    print(f"Total generation time: {(end_all - start_all):.3f}s")


    # --- Denormalize generated spectra and fix shape to [N, 6000]
    for label_name, spectra in generated_spectra.items():
        arr = denormalize_spectra(spectra)
        # arr: [N, 1, 6000] -> [N, 6000]
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        generated_spectra[label_name] = arr


    # --- PIKE matrix ---
    print("\n--- Computing PIKE matrix for generated spectra ---")
    calculate_pike_matrix(generated_spectra, mean_spectra_train, label_correspondence, device, results_path=results_path, saving=True)


    # --- Plot only one generated spectrum per label against mean spectra ---
    for label_name, spectra in generated_spectra.items():
        # Select one generated sample (first sample)
        sample = spectra[0]
        save_path = os.path.join(plots_path, f"{label_name}_vs_all_means.png")
        plot_generated_vs_all_means(sample, mean_spectra_train, label_correspondence, save_path)

