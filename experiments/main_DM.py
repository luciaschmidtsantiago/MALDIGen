import os
import sys
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.DM import ContextUnet1D, generate_spectra_per_label_ddpm
from utils.training_utils import setuplogging, get_and_log, perturb_input
from dataloader.data import load_data, get_dataloaders, compute_summary_spectra_per_label
from losses.PIKE_GPU import calculate_pike_matrix
from utils.visualization import plot_generated_vs_all_means, save_training_curve, plot_generated_spectra_per_label
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
    p.add_argument('--config', default='configs/dm_S.yaml', type=str)
    p.add_argument('--train', action='store_true', default=False, help='Run training')
    p.add_argument('--evaluation', action='store_true', default=False, help='Run evaluation')
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
    plots_path = os.path.join(results_path, 'plots')
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
        device = torch.device('cuda')
    else:
        logger.info("CUDA not available")
        device = torch.device('cpu')
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
        logger.info("STARTING TRAINING LOOP (Early Stopping)")
        nn_model.train()
        training_start = time.time()
        max_patience = get_and_log('max_patience', 10, config, logger)
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = -1
        best_model_path = os.path.join(results_path, "context_model_best.pth")
        train_loss_list = []
        val_loss_list = []
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

            train_loss = np.mean(epoch_losses)
            train_loss_list.append(train_loss)
            logger.info(f"Epoch {ep+1}/{n_epoch} — Train MSE: {train_loss:.6f}")

            # Validation loss (use val_loader)
            nn_model.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, c_val in val_loader:
                    x_val = x_val.to(device)
                    c_val = c_val.float().to(device)
                    mask_val = torch.bernoulli(torch.zeros(c_val.shape[0], device=device) + 0.9)
                    c_val = c_val * mask_val.unsqueeze(-1)
                    noise_val = torch.randn_like(x_val, device=device)
                    t_val = torch.randint(1, timesteps + 1, (x_val.shape[0],), device=device)
                    x_pert_val = perturb_input(x_val, t_val, noise_val, ab_t)
                    t_norm_val = (t_val.float() / timesteps).view(-1, 1)
                    pred_noise_val = nn_model(x_pert_val, t_norm_val, c=c_val)
                    val_loss = F.mse_loss(pred_noise_val, noise_val).item()
                    val_losses.append(val_loss)
            val_loss_mean = np.mean(val_losses)
            val_loss_list.append(val_loss_mean)
            logger.info(f"Epoch {ep+1}/{n_epoch} — Val MSE: {val_loss_mean:.6f}")
            nn_model.train()

            # Early stopping logic
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                patience_counter = 0
                best_epoch = ep
                torch.save(nn_model.state_dict(), best_model_path)
                logger.info(f"  Saved new best model at epoch {ep+1} with val loss {val_loss_mean:.6f}")
            else:
                patience_counter += 1
                logger.info(f"  No improvement. Patience: {patience_counter}/{max_patience}")
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {ep+1}. Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                break

        # Get training time
        training_time = time.time() - training_start
        metadata = {
            "training_time_sec": training_time,
            "time_per_epoch_sec": training_time / max(best_epoch + 1, 1),
            "epochs": best_epoch + 1,
            "total_params": num_params
        }
        config['metadata'] = metadata
        config['pretrained_model_path'] = best_model_path
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
            yaml.dump(metadata, f)
        
        save_training_curve(train_loss_list, val_loss_list, results_path)


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

    if args.evaluation:
        logger.info("EVALUATION MODE")
        nn_model.eval()
        with torch.no_grad():
            val_losses = []
            for x_val, c_val in val_loader:
                x_val = x_val.to(device)
                c_val = c_val.float().to(device)
                noise_val = torch.randn_like(x_val, device=device)
                t_val = torch.randint(1, timesteps + 1, (x_val.shape[0],), device=device)
                x_pert_val = perturb_input(x_val, t_val, noise_val, ab_t)
                t_norm_val = (t_val.float() / timesteps).view(-1, 1)
                pred_noise_val = nn_model(x_pert_val, t_norm_val, c=c_val)
                val_loss = F.mse_loss(pred_noise_val, noise_val).item()
                val_losses.append(val_loss)
            val_loss_mean = np.mean(val_losses)
            logger.info(f"Validation MSE: {val_loss_mean:.6f}")

        with torch.no_grad():
            test_losses = []
            for x_test, c_test in test_loader:
                x_test = x_test.to(device)
                c_test = c_test.float().to(device)
                noise_test = torch.randn_like(x_test, device=device)
                t_test = torch.randint(1, timesteps + 1, (x_test.shape[0],), device=device)
                x_pert_test = perturb_input(x_test, t_test, noise_test, ab_t)
                t_norm_test = (t_test.float() / timesteps).view(-1, 1)
                pred_noise_test = nn_model(x_pert_test, t_norm_test, c=c_test)
                test_loss = F.mse_loss(pred_noise_test, noise_test).item()
                test_losses.append(test_loss)
            test_loss_mean = np.mean(test_losses)
            logger.info(f"Test MSE: {test_loss_mean:.6f}")

    # ============================================================
    # SAMPLE GENERATION
    # ============================================================
    if args.generation:
        logger.info("SPECTRA GENERATION")
        
        label_correspondence = train.label_convergence
        label_ids = sorted(label_correspondence.keys())
        label_names = [label_correspondence[i] for i in label_ids]
        logger.info(f"Generating conditional samples for {len(label_names)} labels: {label_names}")

        # --- Compute mean and std spectra from train set ---
        train_loader_idx = []
        for x, y in train_loader:
            if y.ndim > 1:
                y = y.argmax(dim=1)
            train_loader_idx.append((x, y))
        summary_mean_spectra = compute_summary_spectra_per_label(train_loader_idx, device)

        # --- Denormalize mean and std spectra from [-1,1] → [0,1] and fix shape to [1, 6000]
        for k in summary_mean_spectra:
            mean_arr, std_arr, max_arr, min_arr = summary_mean_spectra[k]
            mean_arr = denormalize_spectra(mean_arr).squeeze(1)
            std_arr = (std_arr/2).squeeze(1)
            max_arr = denormalize_spectra(max_arr).squeeze(1)
            min_arr = denormalize_spectra(min_arr).squeeze(1)
            summary_mean_spectra[k] = (mean_arr, std_arr, max_arr, min_arr)

        # --- Generate spectra per label (Diffusion) ---
        start_all = time.time()
        generated_spectra = generate_spectra_per_label_ddpm(nn_model, label_correspondence, args.n_generate, timesteps, a_t, b_t, ab_t, logger, device)
        end_all = time.time()
        total_time = end_all - start_all
        total_samples = sum(v.shape[0] for v in generated_spectra.values()) if isinstance(generated_spectra, dict) else 0
        per_sample = total_time / total_samples if total_samples > 0 else float('nan')
        logger.info(f"Total generation time: {total_time:.3f}s — {per_sample:.6f}s per sample ({total_samples} samples)")
        metadata['avg_reconstruction_time'] = 0
        metadata['avg_generation_time'] = per_sample

        # Update metadata and save to config
        config['metadata'] = metadata
        with open(args.config, 'w') as f:
            yaml.dump(config, f)

        # --- Denormalize generated spectra and fix shape to [N, 6000]
        for label_name, spectra in generated_spectra.items():
            arr = denormalize_spectra(spectra).squeeze(1)
            # arr: [N, 1, 6000] -> [N, 6000]
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr.squeeze(1)
            generated_spectra[label_name] = arr

        # --- PIKE matrix ---
        logger.info("Computing PIKE matrix for generated spectra")
        # For PIKE, use only the mean part of mean_std_spectra
        mean_spectra_only = {k: v[0] for k, v in summary_mean_spectra.items()}
        calculate_pike_matrix(generated_spectra, mean_spectra_only, label_correspondence, device, results_path=results_path, saving=True)

        # --- Plot only one generated spectrum per label against mean and std spectra ---
        for label_name, spectra in generated_spectra.items():
            # Select one generated sample (first sample)
            sample = spectra[0]
            save_path = os.path.join(plots_path, f"{label_name}")
            plot_generated_vs_all_means(sample, summary_mean_spectra, label_correspondence, save_path, logger)

            # --- Call plot_generated_spectra_per_label for each label ---
            n_samples_to_plot = 5  # or any number you want to plot per label
            name_to_index = {v: k for k, v in label_correspondence.items()}
            try:
                mean_std_minmax = summary_mean_spectra[name_to_index[label_name]]
            except KeyError:
                raise KeyError(f"Label name {label_name} not found in label_correspondence")
            plot_generated_spectra_per_label(spectra, mean_std_minmax, label_name, n_samples_to_plot, plots_path)

    # Append to CSV
    write_metadata_csv(metadata, config, name)

if __name__ == "__main__":
    main()