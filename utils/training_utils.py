import os
import time
import torch
import logging
import numpy as np
import warnings
import torch.nn as nn

from visualization.visualization import save_training_curve, plot_nll_vs_kl, training_curve_gan
from models.VAE import ConditionalVAE

def get_and_log(param, default, config, logger, name=None):
    param_value = config.get(param, default)
    if name is None:
        logger.info(f"{param}: {param_value}")
    else:
        logger.info(f"{name}: {param_value}")
    return param_value

def setuplogging(name, mode, results_path):
    log_file = os.path.join(results_path, f"{mode}_{name}.log")
    os.makedirs(results_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()])
    logger = logging.getLogger('logger')
    return logger


######## VAE TRAINING ########
def run_experiment(model, train_loader, val_loader, config, results_path, logger):

    # TRAINING
    logger.info("=" * 80)
    logger.info(f"TRAINING...:")
    logger.info("=" * 80)

    t_start = time.time()
    [nll_train, nll_val], [kl_train, kl_val], best_model, e = training(
        max_patience=config['max_patience'],
        num_epochs=config['epochs'],
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=config['lr']),
        training_loader=train_loader,
        val_loader=val_loader,
        scheduler=None,
        logger=logger
    )
    training_time = time.time() - t_start

    # Infer epochs from returned losses (expected list-like)
    if isinstance(nll_train, (list, tuple, np.ndarray)):
        epochs_used = len(nll_train)
    else:
        epochs_used = config.get('num_epochs', 0)
    time_per_epoch = training_time / max(epochs_used, 1)
    num_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)

    # METADATA
    metadata = {
        'training_time_sec': training_time,
        'time_per_epoch_sec': time_per_epoch,
        'epochs': epochs_used,
        'num_params': num_params}

    # Plot the training curve
    save_training_curve(nll_train, nll_val, results_path)
    plot_nll_vs_kl(nll_train, kl_train, results_path)

    return best_model, [nll_train, nll_val], metadata

def training(max_patience, num_epochs, model, optimizer, training_loader, val_loader, scheduler=None, logger=None):
    nll_val = []
    nll_train = []
    kl_val = []
    kl_train = []
    best_nll = 1000
    patience = 0
    device = next(model.parameters()).device
    
    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        train_loss = 0.
        train_kl = 0.
        N = 0.
        for _, batch in enumerate(training_loader):
            x, y_species, *maybe_amr = batch
            x, y_species = x.to(device), y_species.to(device)
            y_amr = maybe_amr[0].to(device) if maybe_amr else None

            is_conditional = isinstance(model, ConditionalVAE)
            if is_conditional:
                loss, kl = model.forward(x, y_species, y_amr)
            else:
                loss, kl = model.forward(x)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item()
            train_kl += kl.item()
            N += 1

        train_loss /= N
        train_kl /= N
        nll_train.append(train_loss)
        kl_train.append(train_kl)

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        # Validation
        loss_val, KL_val = evaluation(val_loader, model_best=model)
        nll_val.append(loss_val)
        kl_val.append(KL_val)

        if loss_val < best_nll:
            best_nll = loss_val
            patience = 0
            best_model = model
        else:
            patience = patience + 1

        if e % 5 == 0 or e == num_epochs - 1 or patience > max_patience:
            logger.info(f"Epoch {e}: train nll={train_loss:.4f}, avg KL={train_kl:.4f}, val nll={loss_val:.4f}, patience={patience}")

        if patience > max_patience:
            logger.info(f"Early stopping at epoch {e}.")
            break

    return [nll_train, nll_val], [kl_train, kl_val], best_model, e

def evaluation(test_loader, model_best):
    model_best.eval()
    device = next(model_best.parameters()).device
    loss = 0.
    kl = 0.
    N = 0.
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            x, y_species, *maybe_amr = batch
            x, y_species = x.to(device), y_species.to(device)
            y_amr = maybe_amr[0].to(device) if maybe_amr else None

            from models.VAE import ConditionalVAE
            if isinstance(model_best, ConditionalVAE):
                loss_batch, KL = model_best.forward(x, y_species, y_amr)
            else:
                loss_batch, KL = model_best.forward(x)

            # Convert from summed batch loss to per-sample average
            loss += loss_batch.item()
            kl   += KL.item()
            N += 1

    if N == 0:
        warnings.warn("evaluation(): received empty data loader (N=0). Returning +inf for loss and KL.")
        return float('inf'), float('inf')

    loss /= N
    kl   /= N
    return loss, kl


######## GAN TRAINING ########
def training_gan(model, train_loader, val_loader, criterion, optimizer_G, optimizer_D, device, config, logger=None, weighted=None):
    """
    Train a GAN (both generator and discriminator) with early stopping based on validation loss.
    Args:
        model (nn.Module): GAN model with .forward_G and .forward_D methods.
        train_loader (DataLoader): training dataloader.
        val_loader (DataLoader): validation dataloader.
        criterion (loss): typically nn.BCEWithLogitsLoss().
        optimizer_G (torch.optim): optimizer for generator.
        optimizer_D (torch.optim): optimizer for discriminator.
        device (torch.device): cuda or cpu.
        config (dict): training configuration with keys:
            - 'latent_dim': dimension of latent noise vector.
            - 'epochs': max number of epochs.
            - 'max_patience': epochs to wait for improvement before stopping.
        logger (logging.Logger, optional): logger for info messages.
        weighted (int, optional): number of classes for weighted training. If None, no weighting is applied.
    Returns:
        generator (nn.Module): trained generator model.
        discriminator (nn.Module): trained discriminator model.
        history (dict): training history with keys:
            - "D_train": list of discriminator training losses
            - "G_train": list of generator training losses
            - "D_val": list of discriminator validation losses
            - "G_val": list of generator validation losses
    """

    # Compute class weights (once, before training loop)
    if weighted is not None:
        # Count samples per class in train_loader
        class_counts = torch.zeros(weighted, dtype=torch.float32, device=device)
        for _, y_species, *_ in train_loader:
            y_species = y_species if y_species.ndim == 1 else y_species.argmax(dim=1)
            for c in y_species:
                class_counts[c] += 1
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)

    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = config.get('max_patience', 10)
    history = {"D_train": [], "G_train": [], 
               "D_val": [], "G_val": []}
    

    for epoch in range(config['epochs']):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x_real, y_species, *maybe_amr = batch
            x_real, y_species = x_real.to(device), y_species.to(device)
            y_amr = maybe_amr[0].to(device) if maybe_amr else None

            batch_size = x_real.size(0)
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # === Discriminator ===
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            x_fake = model.forward_G(z, y_species, y_amr).detach()
            d_real = model.forward_D(x_real, y_species, y_amr)
            d_fake = model.forward_D(x_fake, y_species, y_amr)

            if weighted is not None and y_species is not None:
                if y_species.ndim > 1:
                    y_idx = y_species.argmax(dim=1)
                else:
                    y_idx = y_species
                weights = class_weights[y_idx].unsqueeze(1)  # [batch, 1]

                d_loss_real = (criterion(d_real, valid) * weights).mean()
                d_loss_fake = (criterion(d_fake, fake) * weights).mean()
                d_loss = d_loss_real + d_loss_fake
            else:
                d_loss = criterion(d_real, valid) + criterion(d_fake, fake)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # === Generator ===
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            x_gen = model.forward_G(z, y_species, y_amr)
            d_gen = model.forward_D(x_gen, y_species, y_amr)
            g_loss = criterion(d_gen, valid)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

        # Save average losses
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches

        # === VALIDATION ===
        val_total, d_val, g_val = evaluation_gan(
            model, 
            val_loader, 
            criterion,
            latent_dim=config['latent_dim'],
            device=device
        )

        history["D_train"].append(avg_d_loss)
        history["G_train"].append(avg_g_loss)
        history["D_val"].append(d_val)
        history["G_val"].append(g_val)

        logger.info(f"[Epoch {epoch+1}] "
               f"Train D={avg_d_loss:.4f}, G={avg_g_loss:.4f} | "
               f"Val D={d_val:.4f}, G={g_val:.4f}")

        # === EARLY STOPPING ===
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_generator = model.generator.state_dict()
            best_discriminator = model.discriminator.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model weights
    model.generator.load_state_dict(best_generator)
    model.discriminator.load_state_dict(best_discriminator)

    return model.generator, model.discriminator, history

def evaluation_gan(model, loader, criterion, latent_dim, device):
    """
    Evaluate a trained GAN (both discriminator and generator losses).
    Args:
        model (nn.Module): GAN model with .forward_G and .forward_D methods.
        loader (DataLoader): dataloader for evaluation.
        criterion (loss): typically nn.BCEWithLogitsLoss().
        latent_dim (int): dimension of latent noise vector.
        device (torch.device): cuda or cpu.
    Returns:
        total_loss (float): sum of discriminator and generator losses.
        d_loss_avg (float): average discriminator loss.
        g_loss_avg (float): average generator loss.
    """
    model.eval()
    d_loss_total, g_loss_total, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in loader:
            x_real, y_species, *maybe_amr = batch
            x_real, y_species = x_real.to(device), y_species.to(device)
            y_amr = maybe_amr[0].to(device) if maybe_amr else None

            batch_size = x_real.size(0)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            z = torch.randn(batch_size, latent_dim, device=device)
            x_fake = model.forward_G(z, y_species, y_amr)

            d_real = model.forward_D(x_real, y_species, y_amr)
            d_fake = model.forward_D(x_fake, y_species, y_amr)

            d_loss = criterion(d_real, valid) + criterion(d_fake, fake)
            g_loss = criterion(d_fake, valid)

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            n_batches += 1

    d_loss_avg = d_loss_total / n_batches
    g_loss_avg = g_loss_total / n_batches
    total_loss = d_loss_avg + g_loss_avg

    return total_loss, d_loss_avg, g_loss_avg

def run_experiment_gan(model, loaders, device, config, results_path, logger, weighted=None):
    """
    Run a full GAN training experiment with training and evaluation.
    Args:
        model (nn.Module): GAN model with .forward_G and .forward_D methods.
        loaders (tuple): (train_loader, val_loader, test_loader).
        device (torch.device): cuda or cpu.
        config (dict): training configuration with keys:
            - 'latent_dim': dimension of latent noise vector.
            - 'epochs': max number of epochs.
            - 'pretrain_epochs': number of epochs to pretrain discriminator.
            - 'lr_d': learning rate for discriminator.
            - 'lr_g': learning rate for generator.
            - 'max_patience': epochs to wait for improvement before stopping.
        results_path (str): directory to save results.
        logger (logging.Logger): logger for info messages.
    Returns:
        generator (nn.Module): trained generator model.
        discriminator (nn.Module): trained discriminator model.
        metadata (dict): training metadata including time and parameter counts.
    """
    
    train_loader, val_loader, test_loader = loaders

    logger.info("=" * 80)
    logger.info(f"TRAINING...:")
    logger.info("=" * 80)


    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=config['lr_g'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=config['lr_d'], betas=(0.5, 0.999))

    t_start = time.time()
    generator, discriminator, history = training_gan(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        device=device,
        config=config,
        logger=logger,
        weighted=weighted
    )
    training_time = time.time() - t_start

    epochs_used = len(history["D_train"])
    time_per_epoch = training_time / max(epochs_used, 1)
    # Count parameters only for generator (or both if you want total)
    num_params_G = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    num_params_D = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    metadata = {
        "training_time_sec": training_time,
        "time_per_epoch_sec": time_per_epoch,
        "epochs": epochs_used,
        "num_params_G": num_params_G,
        "num_params_D": num_params_D,
        "total_params": num_params_G + num_params_D
    }

    training_curve_gan(history, results_path)

    return generator, discriminator, metadata


############## DIFFUSION MODEL TRAINING ########
def perturb_input(x, t, noise, ab_t):
    """
    Perturbs a real image x at timestep t using the DDPM noise schedule ab_t.
    Supports x of shape (B, C, H, W) or (B, C, L) where L=H*W.
    - x: torch.Tensor, shape (B, C, H, W) or (B, C, L)
    - t: torch.Tensor, shape (B,) or int
    - noise: torch.Tensor, same shape as x
    - ab_t: torch.Tensor, cumulative product of alphas, shape (timesteps+1,)
    Returns: perturbed image (torch.Tensor, same shape as x)
    """
    # Get batch size
    B = x.shape[0]
    # Prepare ab for broadcasting
    if isinstance(t, torch.Tensor):
        ab = ab_t[t].view(B, 1, 1, 1) if x.ndim == 4 else ab_t[t].view(B, 1, 1)
    else:
        ab = ab_t[t].view(1, 1, 1, 1) if x.ndim == 4 else ab_t[t].view(1, 1, 1)
    return x * ab.sqrt() + noise * (1 - ab).sqrt()