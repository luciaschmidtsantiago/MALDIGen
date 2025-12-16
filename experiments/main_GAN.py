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

from models.GAN import CNNDecoder1D_Generator, Discriminator, MLPDecoder1D_Generator, GAN, generate_spectra_gan
from dataloader.data import compute_summary_spectra_per_label, load_data, get_dataloaders
from utils.training_utils import get_and_log, run_experiment_gan, setuplogging, evaluation_gan
from visualization.visualization import plot_meanVSgenerated
from utils.test_utils import write_metadata_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/gan_MLP3_32.yaml', type=str)
    p.add_argument('--train', action='store_true', default=True, help='Run training (default: only evaluation/visualization)')
    p.add_argument('--evaluation', action='store_true', default=False, help='Run evaluation after training (default: False)')
    p.add_argument('--generation', action='store_true', default=False, help='Generate and plot spectra after training/evaluation (default: False)')
    p.add_argument('--n_generate', type=int, default=5, help="Number of synthetic spectra to generate per label")
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
    results_path = os.path.join(base_results, 'gan', name)
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    logger = setuplogging(name, mode, results_path)
    logger.info(f"Using config file: {args.config}")

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
        device = "cuda:0"
    else:
        logger.info("CUDA not available")
        device = "cpu"
    logger.info(f"Device: {device}")

    # ============================================================
    # DATA LOADING
    # ============================================================
    logger.info("DATA LOADING")
    pickle_marisma = get_and_log('pickle_marisma', "pickles/MARISMa_study.pkl", config, logger)
    pickle_driams = get_and_log('pickle_driams', "pickles/DRIAMS_study.pkl", config, logger)
    batch_size = get_and_log('batch_size', 64, config, logger)
    image_dim = get_and_log('output_dim', 6000, config, logger)
    get_labels = get_and_log('get_labels', True, config, logger)

    train, val, test, ood = load_data(pickle_marisma, pickle_driams, get_labels=get_labels)
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size=batch_size)

    # ============================================================
    # HYPERPARAMETERS
    # ============================================================
    logger.info("HYPERPARAMETER SETTING")
    logger.info("--- Network hyperparameters")
    num_layers = get_and_log('n_layers', 3, config, logger)
    batch_norm = get_and_log('batch_norm', False, config, logger)
    latent_dim = get_and_log('latent_dim', 32, config, logger)
    logger.info("--- Training hyperparameters")
    num_epochs = get_and_log('epochs', 30, config, logger)
    lr_g = get_and_log('lr_g', 2e-4, config, logger)
    lr_d = get_and_log('lr_d', 1e-4, config, logger)
    max_patience = get_and_log('max_patience', 10, config, logger)
    use_dropout = get_and_log('use_dropout', True, config, logger)
    drop_p = get_and_log('dropout_prob', 0.1, config, logger) if use_dropout else None
    weighted = get_and_log('weighted', False, config, logger)
    cond_dim = 0

    # Optional pretrained models for evaluation
    pretrained_generator = get_and_log('pretrained_generator', None, config, logger)
    pretrained_discriminator = get_and_log('pretrained_discriminator', None, config, logger)
    
    # ============================================================
    # MODEL SETUP
    # ============================================================
    logger.info("=" * 80)
    logger.info(f"TRAINING CONFIGURATION:")
    logger.info("=" * 80)

    # Initialize models
    gen_arch = config.get('generator', 'MLP')
    if gen_arch == 'MLP':
        generator = MLPDecoder1D_Generator(latent_dim, num_layers, image_dim, cond_dim=cond_dim, use_bn=batch_norm).to(device)
    elif gen_arch == 'CNN':
        generator = CNNDecoder1D_Generator(latent_dim, image_dim, n_layers=num_layers, cond_dim=cond_dim, use_dropout=use_dropout, dropout_prob=drop_p).to(device)
    discriminator = Discriminator(image_dim, cond_dim=cond_dim, use_bn=batch_norm, use_dropout=use_dropout, dropout_prob=drop_p).to(device)

    model = GAN(generator, discriminator).to(device)

    # === MODEL SUMMARIES ===
    dummy_z = torch.zeros(1, latent_dim).to(device)
    dummy_x = torch.zeros(1, image_dim).to(device)

    try:
        gen_summary = summary(generator, dummy_z, show_input=False, show_hierarchical=False)
    except Exception as e:
        logger.warning(f"Generator summary failed: {e}")
        gen_summary = str(generator)

    try:
        disc_summary = summary(discriminator, dummy_x, show_input=False, show_hierarchical=False)
    except Exception as e:
        logger.warning(f"Discriminator summary failed: {e}")
        disc_summary = str(discriminator)

    logger.info("\nGENERATOR:\n" + gen_summary)
    logger.info("\nDISCRIMINATOR:\n" + disc_summary)

    if args.train:
        # Train the GAN using new GAN-specific experiment runner
        loaders = train_loader, val_loader, test_loader
        generator, discriminator, metadata  = run_experiment_gan(model, loaders, device, config, results_path, logger)
        
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
        logger.info("=" * 80)
        logger.info(f"LOADING PRETRAINED MODEL:")
        logger.info("=" * 80)
        # Load the best model from a previous training
        if pretrained_generator is None or pretrained_discriminator is None:
            raise ValueError("Pretrained model paths must be specified in the config for evaluation mode.")
        model.generator.load_state_dict(torch.load(pretrained_generator, map_location=device))
        model.discriminator.load_state_dict(torch.load(pretrained_discriminator, map_location=device))
        logger.info(f"Loaded pretrained generator from {pretrained_generator}")
        logger.info(f"Loaded pretrained discriminator from {pretrained_discriminator}")

    # ============================================================
    # EVALUATION
    # ============================================================
    if args.evaluation:
        logger.info("=" * 80)
        logger.info("EVALUATION")
        logger.info("=" * 80)

        criterion = torch.nn.BCELoss()
        val_loss, val_d, val_g = evaluation_gan(model, val_loader, criterion, config['latent_dim'], device)
        logger.info(f"Validation loss: {val_loss:.4f} (D={val_d:.4f}, G={val_g:.4f})")

        test_loss, test_d, test_g = evaluation_gan(model, test_loader, criterion, config['latent_dim'], device)
        logger.info(f"Test loss: {test_loss:.4f} (D={test_d:.4f}, G={test_g:.4f})")

    # ============================================================
    # GENERATION
    # ============================================================
    if args.generation:
        logger.info("SPECTRA GENERATION")

        # Generate and plot spectra with respect to the TRAINING SET
        train_loader_idx = []
        for x, y in train_loader:
            if y.ndim > 1:
                y = y.argmax(dim=1)
            train_loader_idx.append((x, y))
        summary_mean_spectra = compute_summary_spectra_per_label(train_loader_idx, device)        
        mean_spectra_only = {k: v[0] for k, v in summary_mean_spectra.items()}
        
        
        
        mean_spectra_list = [mean_spectra_only[i].squeeze(0) for i in range(len(mean_spectra_only))]
        print(f"Mean spectra per label (train set) computed for {len(mean_spectra_list)} labels.")

        n_generate = args.n_generate
        start = time.time()
        generated_spectra = generate_spectra_gan(model, n_generate, latent_dim, device)
        end = time.time()
        total_time = end - start
        total_samples = sum(v.shape[0] for v in generated_spectra.values()) if isinstance(generated_spectra, dict) else 0
        per_sample = total_time / total_samples if total_samples > 0 else float('nan')
        logger.info(f"Total generation time: {total_time:.3f}s — {per_sample:.6f}s per sample ({total_samples} samples)")
        metadata['avg_reconstruction_time'] = 0
        metadata['avg_generation_time'] = per_sample

        # Save plot
        saving_path = os.path.join(plots_path, f'GAN_generated_spectrum_{i+1}.png')
        plot_meanVSgenerated(generated_spectra[0], mean_spectra_list, saving_path)

        # Update metadata and save to config
        config['metadata'] = metadata
        with open(args.config, 'w') as f:
            yaml.dump(config, f)

    # Append to CSV
    write_metadata_csv(metadata, config, name)


if __name__ == "__main__":
    main()