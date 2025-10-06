import os
import csv
import time
import torch
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.visualization import save_training_curve, plot_nll_vs_kl
from models.VAE import ConditionalVAE

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

def setup_train(config, logger):

    D = config['input_dim']
    M = config['latent_dim']
    num_layers = config['n_layers']
    num_heads = config.get('n_heads', 2)
    lr = config.get('lr', 1e-3)
    num_epochs = config['epochs']
    max_patience = config['max_patience']
    batch_size = config['batch_size']
    max_pool = config.get('max_pool', False)
    encoder = config.get('encoder', None)
    decoder = config.get('decoder', None)
    model = config['model']

    # Log configuration
    logger.info("=" * 80)
    logger.info(f"RUNNING {os.path.splitext(os.path.basename(__file__))[0]} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"TRAINING CONFIGURATION:")
    logger.info("=" * 80)
    logger.info(f"\nInput dimension: {D}\nLatent dimension: {M}\nNumber of layers: {num_layers}\nLearning rate: {lr}\nMax epochs: {num_epochs}\nMax patience: {max_patience}\nBatch size: {batch_size}\nMax pool: {max_pool}")

    return D, M, num_layers, num_heads, lr, num_epochs, max_patience, batch_size, max_pool, encoder, decoder, model

def run_experiment(model, train_loader, val_loader, test_loader, config, results_path, logger):

    # TRAINING
    logger.info("=" * 80)
    logger.info(f"TRAINING...:")
    logger.info("=" * 80)

    # Elegir función de entrenamiento dependiendo del tipo de modelo
    is_conditional = isinstance(model, ConditionalVAE)
    training_fn = training_cond if is_conditional else training
    evaluation_fn = evaluation_cond if is_conditional else evaluation

    t_start = time.time()
    [nll_train, nll_val], [kl_train, kl_val], best_model, e = training_fn(
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

    logger.info("=" * 80)
    logger.info(f"EVALUATION:")
    logger.info("=" * 80)

    # Evaluate on val set
    val_loss, kl_loss = evaluation_fn(val_loader, best_model)
    logger.info(f"Validation ELBO loss: {val_loss:.2f}, KL loss: {kl_loss:.2f}")

    # Evaluate on test set
    test_loss, test_kl_loss = evaluation_fn(test_loader, best_model)
    logger.info(f"Test ELBO loss: {test_loss:.2f}, KL loss: {test_kl_loss:.2f}")

    return best_model, [nll_train, nll_val], val_loss, test_loss, metadata


######## VAE TRAINING ########

def training(max_patience, num_epochs, model, optimizer, training_loader, val_loader, scheduler=None, logger=None):
    nll_val = []
    nll_train = []
    kl_val = []
    kl_train = []
    best_nll = 1000
    patience = 0
    start_epoch = 0
    device = next(model.parameters()).device
    
    # Main loop
    for e in range(start_epoch, num_epochs):
        # TRAINING
        model.train()
        train_loss = 0.
        train_kl = 0.
        N = 0.
        for _, batch in enumerate(training_loader):
            batch = batch.to(device)
            loss, kl = model.forward(batch)
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

        if e == 0 and start_epoch == 0:
            best_nll = loss_val
            best_model = model
        else:
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
        for _, test_batch in enumerate(test_loader):
            test_batch = test_batch.to(device)
            loss_t, kl_t = model_best.forward(test_batch)
            batch_size = test_batch.shape[0]

            # Convert summed batch loss to per-sample average
            loss += loss_t.item()
            kl   += kl_t.item()
            N += 1

    loss /= N
    kl   /= N
    return loss, kl


######## CONDITIONAL VAE TRAINING ########

def training_cond(max_patience, num_epochs, model, optimizer, training_loader, val_loader, scheduler=None, logger=None):
    nll_val = []
    nll_train = []
    kl_val = []
    kl_train = []
    best_nll = float('inf')
    patience = 0
    device = next(model.parameters()).device

    for e in range(num_epochs):
        model.train()
        train_loss = 0.
        train_kl = 0.
        N = 0.

        for _, batch in enumerate(training_loader):
            x, y_species, *maybe_amr = batch
            x = x.to(device)
            y_species = y_species.to(device)
            y_amr = maybe_amr[0].to(device) if maybe_amr else None

            loss, KL = model.forward(x, y_species, y_amr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_kl += KL.item()
            N += 1

        train_loss /= N
        train_kl /= N
        nll_train.append(train_loss)
        kl_train.append(train_kl)

        if scheduler is not None:
            scheduler.step()

        # Validation
        loss_val, KL_val = evaluation_cond(val_loader, model_best=model)
        nll_val.append(loss_val)
        kl_val.append(KL_val)

        if loss_val < best_nll:
            best_nll = loss_val
            best_model = model
            patience = 0
        else:
            patience += 1

        if e % 5 == 0 or e == num_epochs - 1 or patience > max_patience:
            logger.info(f"Epoch {e}: train nll={train_loss:.4f}, avg KL={train_kl:.4f}, val nll={loss_val:.4f}, patience={patience}")

        if patience > max_patience:
            logger.info(f"Early stopping at epoch {e}")
            break

    return [nll_train, nll_val], [kl_train, kl_val], best_model, e

def evaluation_cond(test_loader, model_best):
    model_best.eval()
    device = next(model_best.parameters()).device
    loss = 0.
    kl = 0.
    N = 0

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            x, y_species, *maybe_amr = batch
            x = x.to(device)
            y_species = y_species.to(device)
            y_amr = maybe_amr[0].to(device) if maybe_amr else None

            loss_batch, KL = model_best.forward(x, y_species, y_amr)
            batch_size = x.size(0)

            # Convert from summed batch loss to per-sample average
            loss += loss_batch.item()
            kl   += KL.item()
            N += 1

    # Average over batches (same as averaging over all samples)
    loss /= N
    kl   /= N
    return loss, kl