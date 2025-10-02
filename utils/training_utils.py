import os
import csv
import time
import torch
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.visualization import save_training_curve

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
    encoder = config['encoder']
    decoder = config['decoder']
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

    device = next(model.parameters()).device
    # TRAINING
    logger.info("=" * 80)
    logger.info(f"TRAINING...:")
    logger.info("=" * 80)
    t_start = time.time()
    nll_train, nll_val, best_model, epochs_used = training(
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

    logger.info("=" * 80)
    logger.info(f"EVALUATION:")
    logger.info("=" * 80)

    # Evaluate on val set
    val_loss = evaluation(val_loader, best_model)
    logger.info(f"Validation ELBO loss: {val_loss:.2f}")

    # Evaluate on test set
    test_loss = evaluation(test_loader, best_model)
    logger.info(f"Test ELBO loss: {test_loss:.2f}")

    return best_model, [nll_train, nll_val], val_loss, test_loss, metadata

def evaluation(test_loader, model_best):
    # EVALUATION
    model_best.eval()
    device = next(model_best.parameters()).device
    loss = 0.
    N = 0.
    for _, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        loss_t = model_best.forward(test_batch)
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N
    return loss

def training(max_patience, num_epochs, model, optimizer, training_loader, val_loader, scheduler=None, logger=None):
    nll_val = []
    nll_train = []
    best_nll = 1000
    patience = 0
    start_epoch = 0
    device = next(model.parameters()).device
    
    # Main loop

    for e in range(start_epoch, num_epochs):
        # TRAINING
        model.train()
        train_loss = 0.
        N_train = 0
        for _, batch in enumerate(training_loader):
            batch = batch.to(device)
            loss = model.forward(batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
            N_train += batch.shape[0]
        train_loss = train_loss / N_train
        nll_train.append(train_loss)

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model)
        nll_val.append(loss_val)  # save for plotting

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

        if e % 10 == 0 or e == num_epochs - 1 or patience > max_patience:
            logger.info(f"Epoch {e}: train nll={train_loss:.4f}, val nll={loss_val:.4f}, patience={patience}")

        if patience > max_patience:
            logger.info(f"Early stopping at epoch {e}.")
            break

    nll_val = np.asarray(nll_val)
    nll_train = np.asarray(nll_train)

    return nll_train, nll_val, best_model, e


# OUTDATED 

def training_cond(max_patience, num_epochs, model, optimizer, training_loader, val_loader, scheduler=None):
    nll_val = []
    best_nll = 1000
    patience = 0
    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for _, batch in enumerate(training_loader):
            if len(batch) == 3:
                batch_x, labels, label2 = batch
            else:
                batch_x, labels = batch
                label2 = None
            batch_x = batch_x.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)
            if label2 is not None:
                label2 = label2.to(next(model.parameters()).device)
                loss = model.forward(batch_x, labels, label2)
            else:
                loss = model.forward(batch_x, labels, torch.zeros(labels.size(0), dtype=torch.long, device=labels.device))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
        # Validation
        loss_val = evaluation_cond(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting
        if e == 0:
            best_nll = loss_val
            best_model = model
        else:
            if loss_val < best_nll:
                best_nll = loss_val
                patience = 0
                best_model = model
            else:
                patience = patience + 1
        if patience > max_patience:
            break
    nll_val = np.asarray(nll_val)
    # Plot the curve
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.show()
    return nll_val, best_model

def evaluation_cond(test_loader, model_best, epoch=None):
    # EVALUATION
    model_best.eval()
    loss = 0.
    N = 0.
    for _, batch in enumerate(test_loader):
        if len(batch) == 3:
            test_batch, labels, label2 = batch
        else:
            test_batch, labels = batch
            label2 = None
        test_batch = test_batch.to(next(model_best.parameters()).device)
        labels = labels.to(next(model_best.parameters()).device)
        if label2 is not None:
            label2 = label2.to(next(model_best.parameters()).device)
            loss_t = model_best.forward(test_batch, labels, label2)
        else:
            loss_t = model_best.forward(test_batch, labels, torch.zeros(labels.size(0), dtype=torch.long, device=labels.device))
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N
    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')
    return loss