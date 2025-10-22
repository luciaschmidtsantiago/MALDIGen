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

from models.VAE import ConditionalVAE
from models.Networks import MLPEncoder1D, MLPDecoder1D, CNNDecoder1D, CNNEncoder1D, CNNAttenEncoder, CNNAttenDecoder
from dataloader.data import load_data, get_dataloaders
from utils.training_utils import run_experiment, setuplogging, setup_train, evaluation
from utils.test_utils import reconerrorPIKE, write_pike_csv, compute_val_time_metrics, write_metadata_csv
from utils.visualization import plot_latent_tsne_conditional, plot_reconstructions_conditional
from losses.PIKE import calculate_PIKE

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/cvae_MLP3_32.yaml', type=str)
    p.add_argument('--train', action='store_true', default=False, help='Run training (default: only evaluation/visualization)')
    p.add_argument('--pike', action='store_true', default=False, help='Calculate PIKE (default: False)')
    return p.parse_args()

def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    metadata = config.get('metadata', {})
    pretrained_model = config.get('pretrained_model', None)

    # ConditionalVAE (species only, y_amr optional)
    y_species_dim = config['y_species_dim']
    y_embed_dim = config['y_embed_dim']
    y_amr_dim = config.get('y_amr_dim', 0)  # Can be 0 if no AMR info used
    get_labels = config.get('get_labels', False)
    embedding = config.get('embedding', True)
    cond_dim = y_embed_dim + y_amr_dim if embedding else y_species_dim + y_amr_dim

    # Logging
    name = args.config.split('/')[-1].split('.')[0]
    mode = 'training' if args.train else 'evaluation'
    # Ensure results are stored in results/vae/
    vae_results_dir = os.path.join(config['results_dir'], 'vae')
    os.makedirs(vae_results_dir, exist_ok=True)
    results_path = os.path.join(vae_results_dir, name)
    logger = setuplogging(name, mode, results_path)
    logger.info(f"Using config file: {args.config}")   

    # Config
    D, M, num_layers, num_heads, lr, num_epochs, max_patience, batch_size, max_pool, encoder, decoder, model = setup_train(config, logger)
    logger.info(f"Using conditional dimension: {cond_dim} (embedding: {embedding})")

    # Data
    train, val, test, ood = load_data(config['pickle_marisma'], config['pickle_driams'], logger, get_labels)
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set up model
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
    logger.info(f"Training {model.__class__.__name__}...")

    if args.train:
        # Train the model
        best_model, [nll_train, nll_val], val_loss, test_loss, metadata = run_experiment(model, train_loader, val_loader, test_loader, config, results_path, logger)
        logger.info(f"Best {model.__class__.__name__} model obtained from training.")
        pretrained_model = os.path.join(results_path, f"best_model_{name}.pt")
        torch.save(best_model.state_dict(), pretrained_model)

        # Save metadata and pretrained model to config YAML
        config['pretrained_model'] = pretrained_model
        config['metadata'] = metadata
        with open(args.config, 'w') as f:
            yaml.dump(config, f)


    else:
        # Load the best model from a previous training
        best_model = model
        best_model.load_state_dict(torch.load(pretrained_model, map_location=device))
        best_model = best_model.to(device)
        logger.info(f"Loaded trained model from {pretrained_model}")

        logger.info("=" * 80)
        logger.info(f"EVALUATION:")
        logger.info("=" * 80)

        # Evaluate on val set
        val_loss = evaluation(val_loader, best_model)
        logger.info(f"Validation ELBO loss: {val_loss:.2f}")

        # Evaluate on test set
        test_loss = evaluation(test_loader, best_model)
        logger.info(f"Test ELBO loss: {test_loss:.2f}")


    # Plot tSNE for validation and test
    plot_latent_tsne_conditional(best_model, val.data.to(device), val.labels.to(device), results_path, 'val')
    plot_latent_tsne_conditional(best_model, test.data.to(device), test.labels.to(device), results_path, 'test')

    # Plot some reconstructions from val
    plot_reconstructions_conditional(best_model, val, 10, results_path, calculate_PIKE, val.labels, random_state=42)

    if args.pike:
        # Calculate PIKE reconstruction error (mean and per class)
        mean_pike, _, class_pike = reconerrorPIKE(best_model, val_loader, logger, val.labels.to(device))
        metadata['validation_pike'] = float(mean_pike)
        with open(args.config, 'w') as f:
            yaml.dump(config, f)

        # Write PIKE results to CSV
        write_pike_csv(name, class_pike, mean_pike, config, label_order=sorted(np.unique(test.labels)))

    # Save metadata to config YAML
    config['metadata'] = metadata
    with open(args.config, 'w') as f:
        yaml.dump(config, f)

    # Append to CSV: yaml filename and metadata
    write_metadata_csv(metadata, config, name)
        
if __name__ == "__main__":
    main()