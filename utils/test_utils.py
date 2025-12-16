import os
import csv
import torch
import time
import numpy as np
from collections import defaultdict

from losses.PIKE_GPU import calculate_PIKE_gpu

def reconerrorPIKE(model, data_loader, logger=None, labels=None):
    """
    Calculate PIKE reconstruction error for all samples in a data loader, and per class if labels are provided.
    Args:
        model: Trained VAE model (should have encoder and decoder attributes).
        data_loader: DataLoader for the dataset to evaluate.
        logger: Logger object for logging results (optional).
        labels: Optional numpy array or list of labels (same order as data_loader.dataset).
    Returns:
        mean_pike: Mean PIKE reconstruction error over all samples.
        pike_errors: List of PIKE errors for all samples.
        class_pike: Dict mapping class label to average PIKE error (if labels provided, else None).
    """
    
    pike_errors = []
    model.eval()
    class_pike = None

    if labels is not None:
        # Convert numeric labels to string names if possible
        if hasattr(labels, 'dtype'):
            labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
        else:
            labels_np = np.array(labels)
        # If integer, map to string names using label_names
        if np.issubdtype(labels_np.dtype, np.integer) and hasattr(data_loader.dataset, 'label_names'):
            label_names = np.array(data_loader.dataset.label_names)
            labels_str = label_names[labels_np]
        else:
            labels_str = labels_np
        pike_by_class = defaultdict(list)
        sample_idx = 0

    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            device = next(model.parameters()).device
            # Robust batch unpacking for conditional/non-conditional
            if isinstance(batch, (list, tuple)):
                test_batch = batch[0].to(device)
                y_species = batch[1].to(device) if len(batch) > 1 else None
            else:
                test_batch = batch.to(device)
                y_species = None

            # Forward pass
            if y_species is not None:
                mu, logvar = model.encoder(test_batch, y_species)
                z = mu
                x_hat = model.decoder(z, y_species)
            else:
                mu, logvar = model.encoder.forward(test_batch)
                z = mu
                x_hat = model.decoder(z)

            x_true = test_batch.cpu().numpy()
            x_hat = x_hat.cpu().numpy()
            batch_size = x_true.shape[0]

            if labels is not None:
                batch_labels = labels_str[sample_idx:sample_idx+batch_size]

            # Compute PIKE error for each sample in batch
            for i in range(batch_size):
                x_true_tensor = torch.tensor(x_true[i], device=device)
                x_hat_tensor = torch.tensor(x_hat[i], device=device)

                pike_err = calculate_PIKE_gpu(x_true_tensor, x_hat_tensor)
                pike_errors.append(pike_err)

                if labels is not None:
                    pike_by_class[batch_labels[i]].append(pike_err)
            total_samples += batch_size
            logger.info(f"Processed batch {batch_idx+1} of test set.")

            if labels is not None:
                sample_idx += batch_size

    mean_pike = np.mean(pike_errors)
    logger.info(f"Mean PIKE reconstruction error (n={total_samples}): {mean_pike:.4f}")
    if labels is not None:
        class_pike = {str(cls): np.mean(errs) for cls, errs in pike_by_class.items()}
        for cls, avg_pike in class_pike.items():
            logger.info(f"Species {cls}: mean PIKE RE = {avg_pike:.4f} (n={len(pike_by_class[cls])})")
    return mean_pike, pike_errors, class_pike

def compute_val_time_metrics(model, val, config):
    """
    Compute average reconstruction time and average generation time per spectrum.

    Returns:
        avg_recon_time: float (seconds per sample)
        avg_gen_time: float (seconds per sample)
    """
    device = next(model.parameters()).device
    val_data = val.data
    batch_size = config.get('batch_size', 64)

    avg_recon_time = None
    avg_gen_time = None

    # --- Average reconstruction time (batched) ---
    if val_data is not None:
        model.eval()
        n_samples = val_data.shape[0]
        recon_times = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                X = torch.tensor(val_data[i:i+batch_size], dtype=torch.float32).to(device)
                start = time.time()
                if hasattr(model.encoder, "forward"):
                    mu, _ = model.encoder.forward(X)
                else:
                    mu = model.encoder(X)
                    mu = mu[:, :mu.size(1) // 2]  # assume [mu, logvar] if not split
                x_hat = model.decoder(mu)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end = time.time()
                recon_times.append((end - start) / X.shape[0])
        avg_recon_time = np.mean(recon_times) if recon_times else None

    # --- Infer latent_dim from model ---
    latent_dim = getattr(model, "latent_dim", None)
    if latent_dim is None and hasattr(model.encoder, "fc"):
        fc_layers = model.encoder.fc
        for layer in reversed(fc_layers):
            if isinstance(layer, torch.nn.Linear):
                latent_dim = layer.out_features // 2  # assume [mu, logvar]
                break
    if latent_dim is None:
        latent_dim = config.get("latent_dim", 16)

    # --- Average generation time ---
    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim).to(device)
        start = time.time()
        x_gen = model.decoder(z)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.time()
        avg_gen_time = (end - start) / batch_size

    return avg_recon_time, avg_gen_time

def write_metadata_csv(metadata, config, name):
    """
    Append metadata to a summary CSV. The first column is the yaml filename, then metadata values.
    Args:
        metadata: dict of metrics/values
        config: config dict (must contain 'results_dir')
        name: name to the yaml file
    """
    csv_summary_path = os.path.join(config['results_dir'], 'summary_experiments.csv')
    file_exists = os.path.exists(csv_summary_path)
    with open(csv_summary_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write header
            writer.writerow(['yaml_file'] + list(metadata.keys()))
        writer.writerow([name] + [metadata[k] for k in metadata.keys()])

def write_pike_csv(name, class_pike_dict, total_pike, config, label_order=None):
    """
    Write PIKE per-class and total PIKE for a model to a CSV file, appending if exists.
    Args:
        name: Name to identify the model/run.
        class_pike_dict: Dict mapping class label to PIKE value.
        total_pike: Overall mean PIKE value.
        config: Config dict (must contain 'results_dir').
        label_order: Optional list of labels to fix column order.
    """
    # Prepare header and row
    if label_order is None:
        label_order = sorted(class_pike_dict.keys())
    header = ['model'] + [f'PIKE_{lbl}' for lbl in label_order] + ['PIKE_total']
    row = [name] + [class_pike_dict.get(lbl, '') for lbl in label_order] + [total_pike]
    # Write or append
    csv_path = os.path.join(config['results_dir'], 'pike_results.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)