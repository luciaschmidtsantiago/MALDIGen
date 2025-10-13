
# -----------------------------
# Imports and Path Setup
# -----------------------------
import os
import sys
import numpy as np
import torch
import argparse
import yaml

# Add project root to sys.path for custom module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom modules
from models.VAE import ConditionalVAE, generate_spectra_per_label
from models.Networks import MLPEncoder1D, MLPDecoder1D, CNNDecoder1D, CNNEncoder1D
from dataloader.data import load_data, get_dataloaders, compute_mean_spectra_per_label
from utils.training_utils import setuplogging
from utils.visualization import plot_generated_spectra_per_label, plot_joint_tsne, plot_tsne_from_saved

# PIKE matrix and generative metrics
from losses.PIKE_GPU import calculate_pike_matrix
from losses.GenMALDI_Metrics import mmd_pike, jaccard_topk, class_distance, neighbour_distance, save_generative_metrics_csv



# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MALDI generation using pretrained cVAE")
    parser.add_argument('--config', default='configs/cvae_MLP3_32.yaml', type=str, help="Path to config YAML file")
    parser.add_argument('--n_generate', type=int, default=1000, help="Number of synthetic spectra to generate per label")
    return parser.parse_args()


# -----------------------------
# Main Generation Pipeline
# -----------------------------
def main():
    args = parse_args()

    # Load configuration
    config = yaml.safe_load(open(args.config))
    name = os.path.splitext(os.path.basename(args.config))[0]
    mode = 'generation'
    results_path = os.path.join(config['results_dir'], name)
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)

    # Set up logging
    logger = setuplogging(name, mode, results_path)
    logger.info(f"Using config file: {args.config}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load pretrained model path
    pretrained_model = config.get('pretrained_model', None)
    logger.info(f"Loading pretrained model from: {pretrained_model}")

    # Data loading
    get_labels = config.get('get_labels', False)
    batch_size = config.get('batch_size', 128)
    train, val, test, ood = load_data(
        config['pickle_marisma'],
        config['pickle_driams'],
        logger,
        get_labels
    )
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size)

    # Compute mean spectra for each label in test set
    mean_spectra_test, _, _ = compute_mean_spectra_per_label(test_loader, device, logger)

    # Model architecture parameters
    D = config['input_dim']
    M = config['latent_dim']
    embedding = config.get('embedding', True)
    y_species_dim = config.get('y_species_dim', 0)
    y_embed_dim = config.get('y_embed_dim', 0)
    y_amr_dim = config.get('y_amr_dim', 0)
    cond_dim = y_embed_dim + y_amr_dim if embedding else y_species_dim + y_amr_dim
    num_layers = config['n_layers']
    max_pool = config.get('max_pool', False)

    # Build encoder
    encoder_type = config['encoder']
    if encoder_type == 'MLPEncoder1D':
        encoder = MLPEncoder1D(D, num_layers, M, cond_dim=cond_dim).to(device)
    elif encoder_type == 'CNNEncoder1D':
        encoder = CNNEncoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool, cond_dim=cond_dim).to(device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Build decoder
    decoder_type = config['decoder']
    if decoder_type == 'MLPDecoder1D':
        decoder = MLPDecoder1D(M, num_layers, D, cond_dim=cond_dim).to(device)
    elif decoder_type == 'CNNDecoder1D':
        decoder = CNNDecoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool, cond_dim=cond_dim).to(device)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    # Instantiate model
    model_type = config['model']
    if model_type == 'cVAE':
        model = ConditionalVAE(encoder, decoder, y_species_dim, y_embed_dim, y_amr_dim, M, embedding).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load pretrained weights
    ckpt = torch.load(pretrained_model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    logger.info(f"Model {model.__class__.__name__} successfully loaded and set to eval mode.")

    # Label correspondence for mapping
    label_correspondence = train.label_convergence
    logger.info(f"Label correspondence: {label_correspondence}")

    # Number of synthetic spectra to generate per label
    n_generate = args.n_generate
    generated_spectra = generate_spectra_per_label(model, label_correspondence, n_generate, device=device)

    # Compute PIKE matrix (saves CSV if saving=True)
    # Keep only 50 spectra per label for PIKE calculation
    generated_spectra_short = {label: spectra[:50] for label, spectra in generated_spectra.items()}
    calculate_pike_matrix(generated_spectra_short, mean_spectra_test, label_correspondence, device, results_path=results_path, saving=True)

    # Plot generated spectra for each label
    n_plot = 5
    for label_name, spectra in generated_spectra.items():
        # Find the corresponding mean spectrum for this label
        mean_spec = mean_spectra_test[[k for k, v in label_correspondence.items() if v == label_name][0]].cpu().numpy().squeeze()
        plot_generated_spectra_per_label(spectra, mean_spec, label_name, n_plot, plots_path)

    # --- Joint t-SNE for all sets and generated ---
    datasets_dict = {
        'train': train,
        'val': val,
        'test': test,
        'ood': ood
    }
    # Build label_name_to_index mapping from label_correspondence
    label_name_to_index = {v: k for k, v in label_correspondence.items()}
    plot_joint_tsne(model, datasets_dict, generated_spectra, label_name_to_index, n_samples=50, results_path=plots_path, logger=logger)

    # --- Additional t-SNE plots from saved coordinates ---
    tsne_path = os.path.join(plots_path, "joint_tsne_coords.npz")
    # 1. Training vs Synthetic
    plot_tsne_from_saved(tsne_path, domains_to_plot=["train", "synthetic"], results_path=plots_path, logger=logger)
    # 2. OOD vs Training
    plot_tsne_from_saved(tsne_path, domains_to_plot=["ood", "train"], results_path=plots_path, logger=logger)
    # 3. OOD vs Synthetic
    plot_tsne_from_saved(tsne_path, domains_to_plot=["ood", "synthetic"], results_path=plots_path, logger=logger)

    # -----------------------------
    # Calculate generative metrics and save per-label CSV
    # -----------------------------
    logger.info("\n--- Generative Metrics (PIKE, MMD, Jaccard, Class/Neighbor Distance) ---")
    # Ensure all spectra are numpy arrays (on CPU) for metrics
    generated_spectra_short_np = {}
    for label, arr in generated_spectra_short.items():
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        generated_spectra_short_np[label] = arr
    save_generative_metrics_csv(generated_spectra_short_np, mean_spectra_test, results_path)

    


if __name__ == "__main__":
    main()