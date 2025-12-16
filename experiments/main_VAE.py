"""
This is the main script to run experiments for training the VAE for species identificaiton.

Training is done using datasets MARISMa (2018-2023) and DRIAMS (A, B).

Testing is done using MARISMa 2024.

Out-Of-Distribution (OOD) detection is done using the DRIAMS C, D dataset.

"""

import os
import sys
import numpy as np
import torch
import argparse
import yaml
from pytorch_model_summary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.VAE import VAE_Bernoulli, generate_spectra_vae
from models.Networks import MLPEncoder1D, MLPDecoder1D, CNNDecoder1D, CNNEncoder1D, CNNAttenEncoder, CNNAttenDecoder
from dataloader.data import load_data, get_dataloaders
from utils.training_utils import run_experiment, setuplogging, get_and_log, evaluation
from utils.test_utils import reconerrorPIKE, write_pike_csv, compute_val_time_metrics, write_metadata_csv
from visualization.visualization import plot_latent_tsne, plot_reconstructions, plot_tsne_real_vs_synth
from losses.PIKE_GPU import calculate_PIKE_gpu

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/vae_CNN3_8_MxP.yaml', type=str)
    p.add_argument('--train', action='store_true', default=False, help='Run training (default: only evaluation/visualization)')
    p.add_argument('--pike', action='store_true', default=False, help='Calculate PIKE (default: False)')
    p.add_argument('--evaluation', action='store_true', default=False, help='Run evaluation')
    p.add_argument('--generation', action='store_true', default=False, help='Run generation after training/eval')
    p.add_argument('--n_generate', type=int, default=500, help='Number of spectra to generate per label')    
    return p.parse_args()

# -----------------------------
# Main Generation Pipeline
# -----------------------------
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
    results_path = os.path.join(base_results, 'vae', name)
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
    batch_size = get_and_log('batch_size', 128, config, logger)

    train, val, test, ood = load_data(pickle_marisma, pickle_driams, get_labels=True)
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size=batch_size)

    # Log dataset sizes to help diagnose empty-split issues
    logger.info(f"Dataset sizes - train: {len(train)}, val: {len(val)}, test: {len(test)}, ood: {len(ood)}")
    logger.info(f"Dataloader batches - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}, ood: {len(ood_loader)}")

    # ============================================================
    # HYPERPARAMETERS
    # ============================================================
    logger.info("HYPERPARAMETER SETTING")
    D = get_and_log('input_dim', 6000, config, logger)

    logger.info("--- Network hyperparameters")
    M = get_and_log('latent_dim', 8, config, logger)
    num_layers = get_and_log('n_layers', 3, config, logger)
    logger.info("n_heads are only used for attention-based models")
    num_heads = get_and_log('n_heads', 2, config, logger)
    max_pool = get_and_log('max_pool', False, config, logger)
    encoder = get_and_log('encoder', 'MLPEncoder1D', config, logger)
    decoder = get_and_log('decoder', 'MLPDecoder1D', config, logger)
    model = get_and_log('model', 'cVAE', config, logger)

    logger.info("--- Training hyperparameters")
    num_epochs = get_and_log('epochs', 200, config, logger)
    lr = get_and_log('lr', 1e-3, config, logger)

    # ============================================================
    # MODEL SETUP
    # ============================================================
    if encoder == 'MLPEncoder1D':
        encoder = MLPEncoder1D(D, num_layers, M).to(device)
    elif encoder == 'CNNEncoder1D':
        encoder = CNNEncoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool).to(device)
    elif encoder == 'CNNAttenEncoder':
        encoder = CNNAttenEncoder(D, M, num_heads, num_layers).to(device)
    logger.info("\nENCODER:\n" + str(summary(encoder, torch.zeros(1, D).to(device), show_input=False, show_hierarchical=False)))

    if decoder == 'MLPDecoder1D':
        decoder = MLPDecoder1D(M, num_layers, D).to(device)
    elif decoder == 'CNNDecoder1D':
        decoder = CNNDecoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool).to(device)
    elif decoder == 'CNNAttenDecoder':
        decoder = CNNAttenDecoder(D, M, num_heads, num_layers).to(device)
    logger.info("\nDECODER:\n" + str(summary(decoder, torch.zeros(1, M).to(device), show_input=False, show_hierarchical=False)))

    if model == 'VAE_Bernoulli':
        model = VAE_Bernoulli(encoder, decoder, M).to(device)
    logger.info(f"Training {model.__class__.__name__}...")


    if args.train:
        # Train the model from scratch
        best_model, [nll_train, nll_val], metadata = run_experiment(model, train_loader, val_loader, config, results_path, logger)
        logger.info(f"Best {model.__class__.__name__} model obtained from training.")
        pretrained_model = os.path.join(results_path, f"best_model_{name}.pt")
        torch.save(best_model.state_dict(), pretrained_model)
        config['pretrained_model'] = pretrained_model
        config['metadata'] = metadata
        
    else:
        logger.info("LOADING PRETRAINED MODEL")
        pretrained_path = config.get('pretrained_model')
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
        best_model = model
        best_model.load_state_dict(torch.load(pretrained_path, map_location=device))
        best_model = best_model.to(device)
        logger.info(f"Loaded trained model from {pretrained_path}")


    # ============================================================
    # EVALUATION
    # ============================================================
    if args.evaluation:
        logger.info("=" * 80)
        logger.info(f"EVALUATION:")
        logger.info("=" * 80)

        # Evaluate on val set
        val_loss, kl_loss = evaluation(val_loader, best_model)
        logger.info(f"Validation ELBO loss: {val_loss:.2f}, KL loss: {kl_loss:.2f}")

        # Evaluate on test set
        test_loss, test_kl_loss = evaluation(test_loader, best_model)
        logger.info(f"Test ELBO loss: {test_loss:.2f}, KL loss: {test_kl_loss:.2f}")

        # Plot tSNE for validation and test
        plot_latent_tsne(best_model, val.data.to(device), val.labels.to(device), results_path, 'val')
        plot_latent_tsne(best_model, test.data.to(device), test.labels.to(device), results_path, 'test')

        # Plot some reconstructions from val
        plot_reconstructions(best_model, val, 10, results_path, calculate_PIKE_gpu, random_state=None)

        avg_recon_time, avg_gen_time = compute_val_time_metrics(model, val.data.to(device), config)
        logger.info(f"Average reconstruction time per spectrum: {avg_recon_time:.6f} sec")
        logger.info(f"Average generation time per spectrum: {avg_gen_time:.6f} sec")

        
        # Update metadata
        metadata['avg_reconstruction_time'] = avg_recon_time
        metadata['avg_generation_time'] = avg_gen_time
        config['metadata'] = metadata


    if args.pike:
        # Calculate PIKE reconstruction error (mean and per class)
        mean_pike, _, class_pike = reconerrorPIKE(best_model, val_loader, logger, val.labels.to(device))
        # Write PIKE results to CSV
        write_pike_csv(name, class_pike, mean_pike, config, label_order=sorted(np.unique(test.labels)))

    # ============================================================
    # SAMPLE GENERATION
    # ============================================================
    if args.generation:
        logger.info("SPECTRA GENERATION")

        # 1. Generate synthetic spectra using the trained VAE
        generated_spectra = generate_spectra_vae(best_model, args.n_generate, device=device)
        # 2. Call t-SNE visualization for real vs synthetic
        plot_tsne_real_vs_synth(best_model, generated_spectra, train, results_path, n_real_per_label=1000, n_synth=args.n_generate)

    # Save metadata to config YAML
    with open(args.config, 'w') as f:
        yaml.dump(config, f)

    # Append to CSV: yaml filename and metadata
    write_metadata_csv(metadata, config, name)
        
if __name__ == "__main__":
    main()