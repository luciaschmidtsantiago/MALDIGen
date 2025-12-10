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
import time
import yaml
from pytorch_model_summary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.VAE import ConditionalVAE, generate_spectra_vae
from models.Networks import MLPEncoder1D, MLPDecoder1D, CNNDecoder1D, CNNEncoder1D
from dataloader.data import load_data, get_dataloaders, compute_summary_spectra_per_label
from utils.training_utils import get_and_log, run_experiment, setuplogging, evaluation
from utils.test_utils import reconerrorPIKE, write_pike_csv, write_metadata_csv
from utils.visualization import plot_latent_tsne_conditional, plot_reconstructions_conditional, plot_generated_vs_all_means, plot_generated_spectra_per_label, plot_joint_tsne, plot_tsne_from_saved
from losses.PIKE_GPU import calculate_PIKE_gpu, calculate_pike_matrix

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/cvae_MLP3_32.yaml', type=str)
    p.add_argument('--train', action='store_true', default=False, help='Run training (default: only evaluation/visualization)')
    p.add_argument('--finetuning', action='store_true', default=False, help='If set, run in finetuning mode (freeze encoder, load pretrained model)')
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
    if args.finetuning:
        mode = 'finetuning'
    else:
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

    # ============================================================
    # HYPERPARAMETERS
    # ============================================================
    logger.info("HYPERPARAMETER SETTING")
    D = get_and_log('input_dim', 6000, config, logger)

    logger.info("--- Network hyperparameters")
    M = get_and_log('latent_dim', 32, config, logger)
    num_layers = get_and_log('n_layers', 2, config, logger)
    logger.info("n_heads are only used for attention-based models")
    num_heads = get_and_log('n_heads', 2, config, logger)
    max_pool = get_and_log('max_pool', False, config, logger)
    encoder = get_and_log('encoder', 'MLPEncoder1D', config, logger)
    decoder = get_and_log('decoder', 'MLPDecoder1D', config, logger)
    model = get_and_log('model', 'cVAE', config, logger)

    logger.info("--- Training hyperparameters")
    num_epochs = get_and_log('epochs', 200, config, logger)
    lr = get_and_log('lr', 1e-3, config, logger)

    # ConditionalVAE (species only, y_amr optional)
    logger.info("--- Conditional hyperparameters")
    y_species_dim = get_and_log('y_species_dim', 6, config, logger)
    y_embed_dim = get_and_log('y_embed_dim', 8, config, logger)
    y_amr_dim = get_and_log('y_amr_dim', 0, config, logger)  # Can be 0 if no AMR info used
    get_labels = get_and_log('get_labels', True, config, logger)
    embedding = get_and_log('embedding', False, config, logger)
    cond_dim = y_embed_dim + y_amr_dim if embedding else y_species_dim + y_amr_dim
    logger.info(f"Using conditional dimension: {cond_dim} (embedding: {embedding})")

    # ============================================================
    # MODEL SETUP
    # ============================================================
    logger.info("MODEL SETUP")
    if encoder == 'MLPEncoder1D':
        encoder = MLPEncoder1D(D, num_layers, M, cond_dim=cond_dim).to(device)
        encoder_input = torch.zeros(1, D + cond_dim).to(device) if cond_dim > 0 else torch.zeros(1, D).to(device)
        logger.info("\nENCODER:\n" + str(summary(encoder, encoder_input, show_input=False, show_hierarchical=False)))
    elif encoder == 'CNNEncoder1D':
        encoder = CNNEncoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool, cond_dim=cond_dim).to(device)

    if decoder == 'MLPDecoder1D':
        decoder = MLPDecoder1D(M, num_layers, D, cond_dim=cond_dim).to(device)
        decoder_input = torch.zeros(1, M + cond_dim).to(device) if cond_dim > 0 else torch.zeros(1, M).to(device)
        logger.info("\nDECODER:\n" + str(summary(decoder, decoder_input, show_input=False, show_hierarchical=False)))
    elif decoder == 'CNNDecoder1D':
        decoder = CNNDecoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool, cond_dim=cond_dim).to(device)

    if model == 'cVAE':
        model = ConditionalVAE(encoder, decoder, y_species_dim, y_embed_dim, y_amr_dim, M, embedding).to(device)
    logger.info(f"MODEL: {model.__class__.__name__}")

    # Assert that --train and --finetuning are mutually exclusive
    if args.train and args.finetuning:
        raise ValueError("Arguments --train and --finetuning cannot both be set to True at the same time.")

    if args.finetuning:
        # Load pretrained model weights except for embedding layers, and freeze only embeddings
        logger.info("LOADING PRETRAINED MODEL (EXCLUDING EMBEDDINGS) & FREEZING EMBEDDINGS")
        pretrained_path = config.get('pretrained_model')
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        # Remove embedding weights from state_dict
        for key in ['encoder.y_embed.weight', 'decoder.y_embed.weight', 'prior.y_embed.weight']:
            if key in state_dict:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        # Freeze only the embedding layers
        if hasattr(model.encoder, "y_embed"):
            for param in model.encoder.y_embed.parameters():
                param.requires_grad = False
        if hasattr(model.decoder, "y_embed"):
            for param in model.decoder.y_embed.parameters():
                param.requires_grad = False
        if hasattr(model, "prior") and hasattr(model.prior, "y_embed"):
            for param in model.prior.y_embed.parameters():
                param.requires_grad = False
        logger.info(f"Loaded pretrained model (excluding embeddings) and frozen embeddings from: {pretrained_path}")
        # Fine-tune decoder and new embeddings
        best_model, [nll_train, nll_val], metadata = run_experiment(model, train_loader, val_loader, test_loader, config, results_path, logger)
        logger.info(f"Best {model.__class__.__name__} model obtained from fine-tuning.")
        finetuned_model_path = os.path.join(results_path, f"finetuned_model_{name}.pt")
        torch.save(best_model.state_dict(), finetuned_model_path)
        config['pretrained_model'] = finetuned_model_path
        config['metadata'] = metadata
        with open(args.config, 'w') as f:
            yaml.dump(config, f)

    else:
        if args.train:
            # Train the model from scratch
            best_model, [nll_train, nll_val], metadata = run_experiment(model, train_loader, val_loader, test_loader, config, results_path, logger)
            logger.info(f"Best {model.__class__.__name__} model obtained from training.")
            pretrained_model = os.path.join(results_path, f"best_model_{name}.pt")
            torch.save(best_model.state_dict(), pretrained_model)
            config['pretrained_model'] = pretrained_model
            config['metadata'] = metadata
            with open(args.config, 'w') as f:
                yaml.dump(config, f)
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
        plot_latent_tsne_conditional(best_model, val.data.to(device), val.labels.to(device), results_path, 'val')
        plot_latent_tsne_conditional(best_model, test.data.to(device), test.labels.to(device), results_path, 'test')

        # Plot some reconstructions from val
        plot_reconstructions_conditional(best_model, val, 10, results_path, calculate_PIKE_gpu, val.labels, random_state=42)

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
        
        label_correspondence = train.label_convergence
        label_ids = sorted(label_correspondence.keys())
        label_names = [label_correspondence[i] for i in label_ids]
        name_to_index = {v: k for k, v in label_correspondence.items()}
        logger.info(f"Generating conditional samples for {len(label_names)} labels: {label_names}")

        # --- Compute mean and std spectra from train set ---
        train_loader_idx = []
        for x, y in train_loader:
            if y.ndim > 1:
                y = y.argmax(dim=1)
            train_loader_idx.append((x, y))
        summary_mean_spectra = compute_summary_spectra_per_label(train_loader_idx, device)

        # --- Generate spectra per label ---
        start_all = time.time()
        generated_spectra = generate_spectra_vae(model, args.n_generate, device, label_correspondence)
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

        # --- PIKE matrix ---
        logger.info("Computing PIKE matrix for generated spectra")
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
            try:
                mean_std_minmax = summary_mean_spectra[name_to_index[label_name]]
            except KeyError:
                raise KeyError(f"Label name {label_name} not found in label_correspondence")
            plot_generated_spectra_per_label(spectra, mean_std_minmax, label_name, n_samples_to_plot, plots_path)


        # ---- PLOT t-SNE for all sets and generated 
        datasets_dict = {
            'train': train,
            'val': val,
            'test': test,
            'ood': ood
        }
        # Build label_name_to_index mapping from label_correspondence
        plot_joint_tsne(model, datasets_dict, generated_spectra, name_to_index, n_samples=50, results_path=plots_path, logger=logger)
        tsne_path = os.path.join(plots_path, "joint_tsne_coords.npz")
        # 1. Training vs Synthetic
        plot_tsne_from_saved(tsne_path, domains_to_plot=["train", "synthetic"], results_path=plots_path, logger=logger)
        # 2. OOD vs Training
        plot_tsne_from_saved(tsne_path, domains_to_plot=["ood", "train"], results_path=plots_path, logger=logger)
        # 3. OOD vs Synthetic
        plot_tsne_from_saved(tsne_path, domains_to_plot=["ood", "synthetic"], results_path=plots_path, logger=logger)


    # Save metadata to config YAML
    config['metadata'] = metadata
    with open(args.config, 'w') as f:
        yaml.dump(config, f)

    # Append to CSV: yaml filename and metadata
    write_metadata_csv(metadata, config, name)
        
if __name__ == "__main__":
    main()