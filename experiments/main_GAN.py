"""
This is the main script to run experiments for training a GAN for synthetic MALDI-TOF spectra generation.

Training is done using datasets MARISMa (2018-2023) and DRIAMS (A, B).
Evaluation can include generating spectra and testing with MARISMa 2024.

"""

from json import decoder
import os
import sys
import torch
import argparse
import yaml
from pytorch_model_summary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.GAN import Discriminator, GAN
from models.VAE import BernoulliDecoder
from models.Networks import MLPDecoder1D, MLPDiscriminator
from dataloader.data import load_data, get_dataloaders
from utils.training_utils import run_experiment, setuplogging, setup_train, evaluation
from utils.test_utils import write_metadata_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/gan_MLP3_8.yaml', type=str)
    p.add_argument('--train', action='store_true', default=False, help='Run training (default: only evaluation/visualization)')
    return p.parse_args()

def main():
    args = parse_args()
    args.train = True
    config = yaml.safe_load(open(args.config))

    metadata = config.get('metadata', {})
    pretrained_generator = config.get('pretrained_generator', None)
    pretrained_discriminator = config.get('pretrained_discriminator', None)

    # Logging
    name = args.config.split('/')[-1].split('.')[0]
    mode = 'training' if args.train else 'evaluation'
    results_path = os.path.join(config['results_dir'], name)
    logger = setuplogging(name, mode, results_path)
    logger.info(f"Using config file: {args.config}")

    # Config (reuse setup_train for consistency, even if not all args apply to GANs)
    D, M, num_layers, num_heads, lr, num_epochs, max_patience, batch_size, max_pool, _, _, model = setup_train(config, logger)

    # Data
    train, val, test, ood = load_data(config['pickle_marisma'], config['pickle_driams'], logger)
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    dropout = config.get('use_dropout', True)
    dropout_p = config.get('dropout_p', 0.3)

    if config['arch_type'] == 'MLP':
        decoder_net = MLPDecoder1D(M, num_layers, D).to(device)
        discriminator_net = MLPDiscriminator(D, M, num_layers, dropout, dropout_p)

    
    generator = BernoulliDecoder(decoder_net).to(device)
    discriminator = Discriminator(discriminator_net).to(device)
    logger.info("\nDECODER:\n" + str(summary(decoder_net, torch.zeros(1, M).to(device), show_input=False, show_hierarchical=False)))
    logger.info("\nGENERATOR:\n" + str(summary(generator, torch.zeros(1, M).to(device), show_input=False, show_hierarchical=False)))
    logger.info("\nDISCRIMINATOR:\n" + str(summary(discriminator, torch.zeros(1, D).to(device), show_input=False, show_hierarchical=False)))

    if model == 'GAN':
        model = GAN(generator, discriminator, M, lambda_g=1.0, lambda_d=1.0).to(device)
    logger.info(f"Training {model.__class__.__name__}...")

    if args.train:
        # Train the GAN
        best_model, [nll_train, nll_val], val_loss, test_loss, metadata = run_experiment(model, train_loader, val_loader, test_loader, config, results_path, logger)
        logger.info(f"Best {model.__class__.__name__} model obtained from training.")
        
        # Save best generator and discriminator
        gen_path = os.path.join(results_path, f"best_generator_{name}.pt")
        disc_path = os.path.join(results_path, f"best_discriminator_{name}.pt")
        torch.save(best_model.generator.state_dict(), gen_path)
        torch.save(best_model.discriminator.state_dict(), disc_path)

        # Update config
        config['pretrained_generator'] = gen_path
        config['pretrained_discriminator'] = disc_path
        config['metadata'] = metadata
        with open(args.config, 'w') as f:
            yaml.dump(config, f)

    else:
        # Load the best model from a previous training
        model.generator.load_state_dict(torch.load(pretrained_generator, map_location=device))
        model.discriminator.load_state_dict(torch.load(pretrained_discriminator, map_location=device))
        model = model.to(device)
        logger.info(f"Loaded pretrained generator from {pretrained_generator}")
        logger.info(f"Loaded pretrained discriminator from {pretrained_discriminator}")


        logger.info("=" * 80)
        logger.info(f"EVALUATION:")
        logger.info("=" * 80)

        # Example: Generate samples for visualization
        # z = torch.randn(16, M).to(device)
        # fake_spectra = best_model.generator(z).detach().cpu()
        # plot_generated_spectra(fake_spectra, results_path, "generated_samples")


    # Update metadata and save to config
    config['metadata'] = metadata
    with open(args.config, 'w') as f:
        yaml.dump(config, f)

    # Append to CSV
    write_metadata_csv(metadata, config, name)


if __name__ == "__main__":
    main()