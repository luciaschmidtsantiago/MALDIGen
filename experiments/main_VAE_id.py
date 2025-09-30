"""
This is the main script to run experiments for training the VAE for species identificaiton.

Training is done using datasets MARISMa (2018-2023) and DRIAMS (A, B).

Testing is done using MARISMa 2024.

Out-Of-Distribution (OOD) detection is done using the DRIAMS C, D dataset.

"""

import os
import sys
import logging
import numpy as np
import torch
import pickle
import argparse

from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from pytorch_model_summary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.VAE import VAE_Bernoulli
from models.cVAE import ConditionalVAE
from utils.training_utils import setuplogging, training, evaluation, reconerrorPIKE, write_pike_csv
from utils.visualization import plot_latent_tsne, plot_reconstructions
from losses.PIKE import pike_reconstruction_error
from models.Networks import MLPEncoder1D, MLPDecoder1D, CNNDecoder1D, CNNEncoder1D, CNNAttenEncoder, CNNAttenDecoder

class MALDI(Dataset):
    def __init__(self, data, labels, normalization=True):
        self.data = data
        self.labels = labels
        self.normalization = normalization

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrum = self.data[idx]
        label = self.labels[idx]
        spectrum = torch.tensor(spectrum, dtype=torch.float32)  # Ensure tensor
        if self.normalization:
            min_val = spectrum.min()
            max_val = spectrum.max()
            spectrum = (spectrum - min_val) / (max_val - min_val + 1e-8)
        return spectrum  #, label


def load_data(paths, logger):
    """
    Loads pickled data from a single path or a list of paths.
    Returns a list of loaded objects (or a single object if only one path is given).
    """
    pickle_marisma, pickle_driams = paths
    logger.info("=" * 80)
    logger.info(f"DATA CONFIGURATION:")
    logger.info("=" * 80)
    logger.info("Loading dataset: MARISMa and DRIAMS")

    #MARISMa
    with open(pickle_marisma, 'rb') as f:
        data_marisma = pickle.load(f)

    spectra_marisma, labels_marisma, metas_marisma = data_marisma['data'], data_marisma['label'], data_marisma['meta']
    years = np.array([m['year'] for m in metas_marisma])
    train_idx_marisma = np.where((years.astype(int) >= 2019) & (years.astype(int) <= 2023))[0]
    val_idx_marisma = np.where(years.astype(int) == 2018)[0]
    val_dataset = MALDI(spectra_marisma[val_idx_marisma], labels_marisma[val_idx_marisma])
    test_idx_marisma = np.where(years.astype(int) == 2024)[0]
    test_dataset_marisma = MALDI(spectra_marisma[test_idx_marisma], labels_marisma[test_idx_marisma])

    # DRIAMS
    with open(pickle_driams, 'rb') as f:
        data_driams = pickle.load(f)

    spectra_driams, labels_driams, metas_driams = data_driams['data'], data_driams['label'], data_driams['meta']
    hospitals = np.array([m['hospital'] for m in metas_driams])
    train_idx_driams = np.where((hospitals == 'DRIAMS_A') | (hospitals == 'DRIAMS_B'))[0]
    ood_idx_driams = np.where((hospitals == 'DRIAMS_C') | (hospitals == 'DRIAMS_D'))[0]
    ood_dataset_driams = MALDI(spectra_driams[ood_idx_driams], labels_driams[ood_idx_driams])

    # --- Combine for training ---
    train_dataset = np.concatenate((spectra_marisma[train_idx_marisma], spectra_driams[train_idx_driams]), axis=0)
    train_dataset_labels = np.concatenate((labels_marisma[train_idx_marisma], labels_driams[train_idx_driams]), axis=0)
    combined_train_dataset = MALDI(train_dataset, train_dataset_labels)

    logger.info("Training with MARISMa (2018-2023) and DRIAMS (A, B)")
    logger.info(f"----Training dataset size: {len(combined_train_dataset)} samples")
    logger.info("Validation with MARISMa (2018)")
    logger.info(f"----Validation dataset size: {len(val_dataset)} samples")
    logger.info("Testing with MARISMa (2024)")
    logger.info(f"----Test dataset size: {len(test_dataset_marisma)} samples")
    logger.info("OOD detection with DRIAMS (C, D)")
    logger.info(f"----OOD dataset size: {len(ood_dataset_driams)} samples")

    logger.info(f"\nLabels: {np.unique(train_dataset_labels)}\n")

    return combined_train_dataset, val_dataset, test_dataset_marisma, ood_dataset_driams, train_dataset, train_dataset_labels

def setup_train(D, M, hidden_layer_neurons, num_layers, lr, num_epochs, max_patience, encoder, decoder, model, logger):

    logger.info("=" * 80)
    logger.info(f"RUNNING {os.path.splitext(os.path.basename(__file__))[0]} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"TRAINING CONFIGURATION:")
    logger.info("=" * 80)
    logger.info(f"\nInput dimension: {D}\nLatent dimension: {M}\nHidden layer neurons: {hidden_layer_neurons}\nNumber of layers: {num_layers}\nLearning rate: {lr}\nMax epochs: {num_epochs}\nMax patience: {max_patience}")
    logger.info(f"Training {model.__class__.__name__}...")
    logger.info("\nENCODER:\n" + str(summary(encoder, torch.zeros(1, D), show_input=False, show_hierarchical=False)))
    logger.info("\nDECODER:\n" + str(summary(decoder, torch.zeros(1, M), show_input=False, show_hierarchical=False)))

    return D, M, hidden_layer_neurons, lr, num_epochs, max_patience, encoder, decoder, model

def main(name, parameters, logger, results_path, train=True):

    pickle_marisma = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pickles', 'MARISMa_study.pkl')
    pickle_driams = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pickles', 'DRIAMS_study.pkl')

    combined_train_dataset, val_dataset, test_dataset_marisma, ood_dataset_driams, train_dataset, train_dataset_labels = load_data((pickle_marisma, pickle_driams), logger=logger)

    # Create DataLoaders
    batch_size = 128
    logger.info(f"Batch size: {batch_size}")
    training_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_marisma, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_dataset_driams, batch_size=batch_size, shuffle=False)


    model, lr, num_epochs, max_patience = parameters
    model_save_path = os.path.join(results_path, f"best_model_{name}.pt")

    if train:
        logger.info("=" * 80)
        logger.info(f"TRAINING...:")
        logger.info("=" * 80)
        optimizer_bern = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad], lr=lr)
        nll_train, nll_val, best_model_bern = training(
            max_patience=max_patience,
            num_epochs=num_epochs,
            model=model,
            optimizer=optimizer_bern,
            training_loader=training_loader,
            val_loader=val_loader,
            scheduler=None,
            results_path=results_path,
            logger=logger
        )
        torch.save(best_model_bern.state_dict(), model_save_path)
        logger.info(f"Best {model.__class__.__name__} model saved to: {model_save_path}")
    else:
        logger.info("=" * 80)
        logger.info(f"LOADING PRE-SAVED MODEL: {model.__class__.__name__} model from: {model_save_path}")
        logger.info("=" * 80)
        best_model_bern = model
        best_model_bern.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    logger.info("=" * 80)
    logger.info(f"TESTING:")
    logger.info("=" * 80)

    test_loss_bern = evaluation(test_loader=test_loader, model_best=best_model_bern)
    logger.info(f"Test ELBO loss (Bernoulli): {test_loss_bern:.2f}")

    # Plot tSNE for testing
    plot_latent_tsne(best_model_bern, test_dataset_marisma.data, test_dataset_marisma.labels, results_path)

    # Plot some reconstructions
    plot_reconstructions(best_model_bern, test_dataset_marisma, 10, results_path, pike_reconstruction_error, random_state=42)

    # Calculate PIKE reconstruction error (mean and per class)
    mean_pike_bern, _, class_pike = reconerrorPIKE(best_model_bern, test_loader, logger=logger, labels=test_dataset_marisma.labels)
    # Write PIKE results to CSV
    csv_path = os.path.join(os.path.dirname(results_path), 'pike_results.csv')
    label_order = sorted(np.unique(test_dataset_marisma.labels))
    write_pike_csv(model_save_path, class_pike, mean_pike_bern, csv_path, label_order=label_order)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train or evaluate VAE for MALDI-TOF species identification.")
    parser.add_argument('--train', action='store_true', help='Run training (default: only evaluation/visualization)')
    parser.add_argument('--D', type=int, default=6000, help='Input dimension (default: 6000)')
    parser.add_argument('--M', type=int, default=8, help='Latent dimension (default: 8)')
    parser.add_argument('--hidden_layer_neurons', type=int, default=128, help='Number of neurons per hidden layer (default: 128)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of hidden layers for MLP or conv layers for CNN (default: 4 for MLP, 3 for CNN)')
    parser.add_argument('--max_pool', action='store_true', help='Use max pooling (default: False)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs (default: 5)')
    parser.add_argument('--max_patience', type=int, default=20, help='Early stopping patience (default: 20)')
    parser.add_argument('--encoder', type=str, default='MLPEncoder1D', help='Encoder type (default: MLPEncoder1D)')
    parser.add_argument('--decoder', type=str, default='MLPDecoder1D', help='Decoder type (default: MLPDecoder1D)')
    parser.add_argument('--model', type=str, default='VAE_Bernoulli', help='Model type (default: VAE_Bernoulli)')

    args = parser.parse_args()

    # For MLP, use num_layers and hidden_layer_neurons to build hidden_layer_sizes
    if args.encoder == 'MLPEncoder1D':
        encoder = MLPEncoder1D(args.D, args.num_layers, args.hidden_layer_neurons, args.M)
    elif args.encoder == 'CNNEncoder1D':
        encoder = CNNEncoder1D(args.M, (1, args.D), enc_hidden2=args.hidden_layer_neurons, num_layers=args.num_layers, max_pool=args.max_pool)
    elif args.encoder == 'CNNAttenEncoder':
        encoder = CNNAttenEncoder(D=args.D, latent_dim=args.M, n_heads=2, n_layers=args.num_layers, token_dim=args.hidden_layer_neurons)

    if args.decoder == 'MLPDecoder1D':
        decoder = MLPDecoder1D(args.M, args.num_layers, args.hidden_layer_neurons, args.D)
    elif args.decoder == 'CNNDecoder1D':
        decoder = CNNDecoder1D(args.M, (1, args.D), dec_hidden2=args.hidden_layer_neurons, num_layers=args.num_layers, max_pool=args.max_pool)
    elif args.decoder == 'CNNAttenDecoder':
        decoder = CNNAttenDecoder(D=args.D, latent_dim=args.M, n_heads=2, n_layers=args.num_layers, token_dim=args.hidden_layer_neurons)

    if args.model == 'VAE_Bernoulli':
        model = VAE_Bernoulli(encoder_net=encoder, decoder_net=decoder, M=args.M)
    # elif args.model == 'cVAE':
    #     model = ConditionalVAE(encoder_net=encoder, decoder_net=decoder, M=args.M)

    parameters = model, args.lr, args.num_epochs, args.max_patience

    name = f"{args.model}_{args.encoder}_{args.decoder}"

    # Set up logging
    manually_added_experiment = '20250929_123616'
    experiment = datetime.now().strftime('%Y%m%d_%H%M%S') if args.train else manually_added_experiment

    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', f"{experiment}")
    logger = setuplogging(name, 'training', results_path) if args.train else setuplogging(name, 'evaluation', results_path)

    if args.train:
        setup_train(args.D, args.M, args.hidden_layer_neurons, args.num_layers, args.lr, args.num_epochs, args.max_patience, encoder, decoder, model, logger=logger)

    main(name=name, parameters=parameters, logger=logger, results_path=results_path, train=args.train)