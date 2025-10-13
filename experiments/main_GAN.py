"""
This is the main script to run experiments for training a GAN for synthetic MALDI-TOF spectra generation.

Training is done using datasets MARISMa (2018-2023) and DRIAMS (A, B).
Evaluation can include generating spectra and testing with MARISMa 2024.

"""

import os
import sys
import torch
import argparse
import yaml
import time
from pytorch_model_summary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.GAN import GenerationNetwork, Discriminator, MLPDecoder1D_Generator
from dataloader.data import compute_mean_spectra_per_label, load_data, get_dataloaders
from utils.training_utils import run_experiment_gan, setuplogging, evaluation_gan
from utils.visualization import plot_gan_meanVSgenerated
from utils.test_utils import write_metadata_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/gan_MLP3_32.yaml', type=str)
    p.add_argument('--train', action='store_true', default=False, help='Run training (default: only evaluation/visualization)')
    return p.parse_args()

def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    metadata = config.get('metadata', {})

    # Logging
    name = args.config.split('/')[-1].split('.')[0]
    mode = 'training' if args.train else 'evaluation'
    results_path = os.path.join(config['results_dir'], name)
    logger = setuplogging(name, mode, results_path)
    logger.info(f"Using config file: {args.config}")

    # Hyperparameters
    batch_size = config.get('batch_size', 128)
    pretrained_generator = config.get('pretrained_generator', None)
    pretrained_discriminator = config.get('pretrained_discriminator', None)
    latent_dim = config.get('latent_dim', 32)
    image_dim = config.get('input_dim', 6000)
    num_epochs = config.get('epochs', 30)
    num_layers = config.get('n_layers', 3)
    batch_norm = config.get('batch_norm', False)
    lr_g = config.get('lr_g', 2e-4)
    lr_d = config.get('lr_d', 1e-4)
    max_patience = config.get('max_patience', 10)

    logger.info(f"TRAINING CONFIGURATION:")
    logger.info("=" * 80)
    logger.info(f"\nInput dimension: {image_dim}\nLatent dimension: {latent_dim}\nLearning rate (G): {lr_g}\nLearning rate (D): {lr_d}\nMax epochs: {num_epochs}\nMax patience: {max_patience}\nBatch size: {batch_size}\nBatch norm: {batch_norm}")

    # Data
    train, val, test, ood = load_data(config['pickle_marisma'], config['pickle_driams'], logger, get_labels=True)
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize models
    # generator = GenerationNetwork(latent_dim, image_dim, batch_norm).to(device)
    generator = MLPDecoder1D_Generator(latent_dim, num_layers, image_dim, cond_dim=0, use_bn=batch_norm).to(device)
    discriminator = Discriminator(image_dim).to(device)

    logger.info("\nGENERATOR:\n" + str(summary(generator, torch.zeros(1, latent_dim).to(device), show_input=False, show_hierarchical=False)))
    logger.info("\nDISCRIMINATOR:\n" + str(summary(discriminator, torch.zeros(1, image_dim).to(device), show_input=False, show_hierarchical=False)))


    if args.train:
        # Train the GAN using new GAN-specific experiment runner
        loaders = train_loader, val_loader, test_loader
        generator, discriminator, metadata = run_experiment_gan(generator, discriminator, loaders, device, config, results_path, logger)

        # Save best models
        gen_path = os.path.join(results_path, 'best_generator.pt')
        disc_path = os.path.join(results_path, 'best_discriminator.pt')
        torch.save(generator.state_dict(), gen_path)
        torch.save(discriminator.state_dict(), disc_path)
        logger.info(f"Saved best generator to {gen_path}")
        logger.info(f"Saved best discriminator to {disc_path}")

        # Update config
        config['pretrained_generator'] = gen_path
        config['pretrained_discriminator'] = disc_path
        config['metadata'] = metadata
        with open(args.config, 'w') as f:
            yaml.dump(config, f)

    else:
        # Load the best model from a previous training
        if pretrained_generator is None or pretrained_discriminator is None:
            raise ValueError("Pretrained model paths must be specified in the config for evaluation mode.")
        generator.load_state_dict(torch.load(pretrained_generator, map_location=device))
        discriminator.load_state_dict(torch.load(pretrained_discriminator, map_location=device))
        logger.info(f"Loaded pretrained generator from {pretrained_generator}")
        logger.info(f"Loaded pretrained discriminator from {pretrained_discriminator}")

        criterion = torch.nn.BCELoss()
        val_loss, val_d, val_g = evaluation_gan(val_loader, generator, discriminator, criterion, config['latent_dim'], device)
        logger.info(f"Validation loss: {val_loss:.4f} (D={val_d:.4f}, G={val_g:.4f})")

        test_loss, test_d, test_g = evaluation_gan(test_loader, generator, discriminator, criterion, config['latent_dim'], device)
        logger.info(f"Test loss: {test_loss:.4f} (D={test_d:.4f}, G={test_g:.4f})")

    # Generate and plot spectra with respect to the TRAINING SET
    mean_spectra_train, _, _ = compute_mean_spectra_per_label(train_loader, device, logger)
    mean_spectra_list = [mean_spectra_train[i].squeeze(0) for i in range(len(mean_spectra_train))]
    print(f"Mean spectra per label (train set) computed for {len(mean_spectra_list)} labels.")

    n_samples = 6
    z = torch.randn(n_samples, latent_dim).to(device)
    gen_times = []
    with torch.no_grad():
        for i in range(n_samples):
            start = time.time()
            sample = generator(z[i].unsqueeze(0)).cpu().squeeze(0)  # [1, image_dim], already passed through sigmoid in generator
            end = time.time()
            gen_times.append(end - start)
            saving_path = os.path.join(results_path, 'plots', f'GAN_generated_spectrum_{i+1}.png')
            os.makedirs(os.path.dirname(saving_path), exist_ok=True)
            plot_gan_meanVSgenerated(sample, mean_spectra_list, saving_path)
    avg_gen_time = sum(gen_times) / n_samples
    metadata['avg_generation_time'] = avg_gen_time
    logger.info(f"Average generation time per spectrum: {avg_gen_time:.6f} sec")

    # Update metadata and save to config
    config['metadata'] = metadata
    with open(args.config, 'w') as f:
        yaml.dump(config, f)

    # Append to CSV
    write_metadata_csv(metadata, config, name)


if __name__ == "__main__":
    main()