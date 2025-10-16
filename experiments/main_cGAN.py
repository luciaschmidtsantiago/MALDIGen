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

from models.GAN import CNNDecoder1D_Generator, Discriminator, MLPDecoder1D_Generator, ConditionalGAN
from dataloader.data import compute_mean_spectra_per_label, load_data, get_dataloaders
from utils.training_utils import run_experiment_gan, setuplogging, evaluation_gan
from utils.visualization import plot_gan_meanVSgenerated
from utils.test_utils import write_metadata_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/cgan_MLP3_32.yaml', type=str)
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
    use_dropout = config.get('use_dropout', True)
    drop_p = config.get('dropout_prob', 0.1)

    # Conditional GAN Hyperparameters
    y_species_dim = config['y_species_dim']
    y_embed_dim = config['y_embed_dim']
    y_amr_dim = config.get('y_amr_dim', 0)  # Can be 0 if no AMR info used
    get_labels = config.get('get_labels', True)
    embedding = config.get('embedding', True)
    cond_dim = y_embed_dim + y_amr_dim if embedding else y_species_dim + y_amr_dim
    logger.info(f"Using conditional dimension: {cond_dim} (embedding: {embedding})")

    logger.info(f"TRAINING CONFIGURATION:")
    logger.info("=" * 80)
    logger.info(f"\nInput dimension: {image_dim}\nLatent dimension: {latent_dim}\nLearning rate (G): {lr_g}\nLearning rate (D): {lr_d}\nMax epochs: {num_epochs}\nMax patience: {max_patience}\nBatch size: {batch_size}\nBatch norm: {batch_norm}\nNumber of layers: {num_layers}")

    # Data
    train, val, test, ood = load_data(config['pickle_marisma'], config['pickle_driams'], logger, get_labels=get_labels)
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize models
    # generator = GenerationNetwork(latent_dim, image_dim, batch_norm).to(device)
    gen_arch = config.get('generator', 'MLP')
    if gen_arch == 'MLP':
        generator = MLPDecoder1D_Generator(latent_dim, num_layers, image_dim, cond_dim=cond_dim, use_bn=batch_norm).to(device)
    elif gen_arch == 'CNN':
        generator = CNNDecoder1D_Generator(latent_dim, image_dim, num_layers=num_layers, cond_dim=cond_dim, use_bn=batch_norm).to(device)
    discriminator = Discriminator(image_dim, cond_dim=cond_dim, use_bn=batch_norm, use_dropout=use_dropout, dropout_prob=drop_p).to(device)

    model = ConditionalGAN(generator=generator,
                        discriminator=discriminator,
                        y_species_dim=y_species_dim,
                        y_embed_dim=y_embed_dim,
                        y_amr_dim=y_amr_dim,
                        embedding=embedding).to(device)

    dummy_z = torch.zeros(1, latent_dim).to(device)
    dummy_x = torch.zeros(1, image_dim).to(device)
    dummy_cond = torch.zeros(1, cond_dim).to(device) if cond_dim > 0 else None

    def safe_summary(model, *args):
        try:
            return summary(model, *[a for a in args if a is not None], show_input=False, show_hierarchical=False)
        except Exception as e:
            logger.warning(f"Summary failed for {model.__class__.__name__}: {e}")
            return str(model)

    gen_summary = safe_summary(generator, dummy_z, dummy_cond)
    disc_summary = safe_summary(discriminator, dummy_x, dummy_cond)

    logger.info("\nGENERATOR:\n" + gen_summary)
    logger.info("\nDISCRIMINATOR:\n" + disc_summary)


    if args.train:
        # Train the GAN using new GAN-specific experiment runner
        loaders = train_loader, val_loader, test_loader
        generator, discriminator, metadata = run_experiment_gan(model, loaders, device, config, results_path, logger)

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
        model.generator.load_state_dict(torch.load(pretrained_generator, map_location=device))
        model.discriminator.load_state_dict(torch.load(pretrained_discriminator, map_location=device))
        logger.info(f"Loaded pretrained generator from {pretrained_generator}")
        logger.info(f"Loaded pretrained discriminator from {pretrained_discriminator}")

        criterion = torch.nn.BCELoss()
        val_loss, val_d, val_g = evaluation_gan(model, val_loader, criterion, config['latent_dim'], device)
        logger.info(f"Validation loss: {val_loss:.4f} (D={val_d:.4f}, G={val_g:.4f})")

        test_loss, test_d, test_g = evaluation_gan(model, test_loader, criterion, config['latent_dim'], device)
        logger.info(f"Test loss: {test_loss:.4f} (D={test_d:.4f}, G={test_g:.4f})")

    # Generate and plot spectra with respect to the TRAINING SET
    mean_spectra_train, _, _ = compute_mean_spectra_per_label(train_loader, device, logger)
    mean_spectra_list = [mean_spectra_train[i].squeeze(0) for i in range(len(mean_spectra_train))]
    print(f"Mean spectra per label (train set) computed for {len(mean_spectra_list)} labels.")

    label_correspondence = train.label_convergence
    label_ids = sorted(label_correspondence.keys())
    label_names = [label_correspondence[i] for i in label_ids]
    logger.info(f"Generating conditional samples for {len(label_names)} labels: {label_names}")

    gen_times = []
    os.makedirs(os.path.join(results_path, 'plots'), exist_ok=True)

    with torch.no_grad():
        for label_id in label_ids:
            label_name = label_correspondence[label_id]

            # Prepare conditional inputs
            y_species = torch.tensor([label_id], dtype=torch.long, device=device)
            y_amr = None  # optional conditioning; keep as None if unused

            # Latent noise vector
            z = torch.randn(1, latent_dim, device=device)

            # Generate sample
            start = time.time()
            sample = model.forward_G(z, y_species, y_amr).squeeze(0)  # [image_dim]
            end = time.time()
            gen_times.append(end - start) 

            # Save plot comparing generated vs mean spectrum
            saving_path = os.path.join(results_path, 'plots', f'GAN_generated_{label_name}.png')
            plot_gan_meanVSgenerated(sample, mean_spectra_list, saving_path)
            logger.info(f"Generated sample for {label_name} → saved to {saving_path}")

    # === Compute and log average generation time ===
    avg_gen_time = sum(gen_times) / len(gen_times)
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