import os
import sys
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.DM import ContextUnet1D, generate_spectra_per_label_ddpm
from utils.training_utils import setuplogging, get_and_log, perturb_input
from dataloader.data import load_data, get_dataloaders, compute_mean_spectra_per_label
from losses.PIKE_GPU import calculate_pike_matrix
from utils.visualization import plot_generated_vs_all_means
from utils.test_utils import write_metadata_csv

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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/dm_default.yaml', type=str)
    p.add_argument('--train', action='store_true', default=False, help='Run training')
    p.add_argument('--generation', action='store_true', default=False, help='Run generation after training/eval')
    p.add_argument('--n_generate', type=int, default=500, help='Number of spectra to generate per label')
    return p.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    metadata = config.get('metadata', {})

    name = args.config.split('/')[-1].split('.')[0]
    mode = 'training' if args.train else 'evaluation'
    
    # ============================================================
    # DIRECTORIES & LOGGING
    # ============================================================
    base_results = config.get('results_dir', 'results')
    results_path = os.path.join(base_results, 'dm', name)
    checkpoints_path = os.path.join(results_path, 'checkpoints')
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    logger = setuplogging(name, mode, results_path)
    logger.info(f"Loaded config: {args.config}")

    # ============================================================
    # DEVICE SETUP
    # ============================================================
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info("CUDA available")
        logger.info(f"   - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"     Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        logger.info(f"   - Current device: {torch.cuda.current_device()}")
        device = "cuda:0"
    else:
        logger.info("CUDA not available")
        device = "cpu"
    logger.info(f"Device: {device}")

    # ============================================================
    # DATA LOADING
    # ============================================================
    logger.info("DATA LOADING")
    pickle_marisma = get_and_log('pickle_marisma', "pickles/MARISMa_study.pkl", config, logger)
    pickle_driams = get_and_log('pickle_driams', "pickles/DRIAMS_study.pkl", config, logger)
    batch_size = get_and_log('batch_size', 64, config, logger)
    output_dim = get_and_log('output_dim', 6000, config, logger)
    n_classes = get_and_log('n_classes', 6, config, logger)

    train, val, test, ood = load_data(pickle_marisma, pickle_driams, get_labels=True, model_type='diffusion')
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size=batch_size)

    for i, batch in enumerate(train_loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            logger.info(f"Batch {i}: x shape={x.shape}, y shape={y.shape}")
            logger.info(f"  x min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
        if i >= 2:
            break

    assert x.shape[2] == output_dim, f"Expected input dimension {output_dim}, got {x.shape[2]}"
    assert y.shape[1] == n_classes, f"Expected {n_classes} classes, got {y.shape[1]}"

    # ============================================================
    # HYPERPARAMETERS
    # ============================================================
    logger.info("HYPERPARAMETER SETTING")
    timesteps = get_and_log('timesteps', 500, config, logger)
    beta1 = get_and_log('beta1', 1e-4, config, logger)
    beta2 = get_and_log('beta2', 0.02, config, logger)
    n_channels = get_and_log('n_channels', 1, config, logger)

    logger.info("--- Network hyperparameters")
    base_features = get_and_log('base_features', 64, config, logger)
    num_blocks = get_and_log('num_blocks', 2, config, logger)
    n_cfeat = get_and_log('n_cfeat', n_classes, config, logger)
    norm_groups = get_and_log('norm_groups', 8, config, logger)
    kernel_size = get_and_log('kernel_size', 4, config, logger)

    logger.info("--- Training hyperparameters")
    n_epoch = get_and_log('n_epoch', 32, config, logger)
    lrate = get_and_log('lrate', 1e-3, config, logger)

    # Diffusion schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device, dtype=torch.float32) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1.0

    # ============================================================
    # MODEL SETUP
    # ============================================================
    logger.info("MODEL SETUP")
    nn_model = ContextUnet1D(
                in_channels=n_channels,
                n_feat=base_features,
                n_cfeat=n_cfeat,
                length=output_dim,
                n_blocks=num_blocks,
                norm_groups=norm_groups,
                kernel_size=kernel_size,
                logger=logger
    ).to(device)

    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
    logger.info(f"Optimizer: {optim}")

    num_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
    metadata['num_params'] = num_params

    # Model summary (optional)
    try:
        from pytorch_model_summary import summary
        logger.info(
            "\n" + summary(
                nn_model,
                torch.zeros((1, n_channels, output_dim), device=device),
                torch.zeros((1, 1), device=device),
                torch.zeros((1, n_cfeat), device=device),
                show_input=True
            )
        )
    except Exception as e:
        logger.warning(f"Could not print model summary: {e}")
    logger.info(f"Total parameters: {sum(p.numel() for p in nn_model.parameters())/1e6:.2f}M")


    # ============================================================
    # TRAINING LOOP
    # ============================================================
    if args.train:
        logger.info("STARTING TRAINING LOOP")
        nn_model.train()
        training_start = time.time()
        for ep in range(n_epoch):
            optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{n_epoch}")

            for x, c in pbar:
                optim.zero_grad()
                x = x.to(device)
                c = c.float().to(device)

                # Mask context
                mask = torch.bernoulli(torch.zeros(c.shape[0], device=device) + 0.9)
                c = c * mask.unsqueeze(-1)

                # Perturb data
                noise = torch.randn_like(x, device=device)
                t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device)
                x_pert = perturb_input(x, t, noise, ab_t)
                t_norm = (t.float() / timesteps).view(-1, 1)

                # Predict noise
                pred_noise = nn_model(x_pert, t_norm, c=c)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optim.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": np.mean(epoch_losses)})

            logger.info(f"Epoch {ep+1}/{n_epoch} — MSE: {np.mean(epoch_losses):.6f}")

            if ep % 4 == 0 or ep == n_epoch - 1:
                ckpt_path = os.path.join(checkpoints_path, f"context_model_{ep}.pth")
                torch.save(nn_model.state_dict(), ckpt_path)
                logger.info(f"Saved model: {ckpt_path}")
        
        # Get training time
        training_time = time.time() - training_start
        metadata['training_time_sec'] = training_time
        metadata['epochs'] = n_epoch
        metadata['time_per_epoch_sec'] = training_time / max(n_epoch, 1)


    # ============================================================
    # LOADING PRETRAINED MODEL
    # ============================================================
    else:
        logger.info("LOADING PRETRAINED MODEL")
        pretrained_path = config.get('pretrained_model_path')
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
        nn_model.load_state_dict(torch.load(pretrained_path, map_location=device))
        logger.info(f"Loaded weights from {pretrained_path}")

    # ============================================================
    # SAMPLE GENERATION
    # ============================================================
    if args.generation:
        logger.info("SPECTRA GENERATION")
        
        label_correspondence = train.label_convergence
        label_ids = sorted(label_correspondence.keys())
        label_names = [label_correspondence[i] for i in label_ids]
        logger.info(f"Generating conditional samples for {len(label_names)} labels: {label_names}")

        # --- Compute mean spectra from train set ---
        train_loader_idx = []
        for x, y in train_loader:
            if y.ndim > 1:
                y = y.argmax(dim=1)
            train_loader_idx.append((x, y))
        # Then pass that temporary loader
        mean_spectra_train, _, _ = compute_mean_spectra_per_label(train_loader_idx, device)


        # --- Denormalize mean spectra from [-1,1] → [0,1] and fix shape to [1, 6000]
        for k in mean_spectra_train:
            arr = denormalize_spectra(mean_spectra_train[k])
            # arr: [1, 1, 6000] -> [1, 6000]
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr.squeeze(1)
            mean_spectra_train[k] = arr

        # --- Generate spectra per label (Diffusion) ---
        n_generate = 500  # For PIKE matrix
        start_all = time.time()
        generated_spectra = generate_spectra_per_label_ddpm(nn_model, label_correspondence, n_generate, timesteps, a_t, b_t, ab_t, logger, device)
        end_all = time.time()
        total_time = end_all - start_all
        total_samples = sum(v.shape[0] for v in generated_spectra.values()) if isinstance(generated_spectra, dict) else 0
        per_sample = total_time / total_samples if total_samples > 0 else float('nan')
        logger.info(f"Total generation time: {total_time:.3f}s — {per_sample:.6f}s per sample ({total_samples} samples)")
        metadata['avg_generation_time'] = per_sample

        # Update metadata and save to config
        config['metadata'] = metadata
        with open(args.config, 'w') as f:
            yaml.dump(config, f)


        # --- Denormalize generated spectra and fix shape to [N, 6000]
        for label_name, spectra in generated_spectra.items():
            arr = denormalize_spectra(spectra)
            # arr: [N, 1, 6000] -> [N, 6000]
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr.squeeze(1)
            generated_spectra[label_name] = arr


        # --- PIKE matrix ---
        logger.info("Computing PIKE matrix for generated spectra")
        calculate_pike_matrix(generated_spectra, mean_spectra_train, label_correspondence, device, results_path=results_path, saving=True)


        # --- Plot only one generated spectrum per label against mean spectra ---
        for label_name, spectra in generated_spectra.items():
            # Select one generated sample (first sample)
            sample = spectra[0]
            save_path = os.path.join(plots_path, f"{label_name}_vs_all_means.png")
            plot_generated_vs_all_means(sample, mean_spectra_train, label_correspondence, save_path, logger)

    # Append to CSV
    write_metadata_csv(metadata, config, name)

if __name__ == "__main__":
    main()