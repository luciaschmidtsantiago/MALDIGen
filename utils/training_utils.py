import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.visualization import save_training_curve
from losses.PIKE import pike_reconstruction_error

def evaluation(test_loader, model_best):
    # EVALUATION
    model_best.eval()
    loss = 0.
    N = 0.
    for _, test_batch in enumerate(test_loader):
        loss_t = model_best.forward(test_batch)
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N
    return loss

def training(max_patience, num_epochs, model, optimizer, training_loader, val_loader, scheduler=None, results_path=None, logger=None):
    nll_val = []
    best_nll = 1000
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for _, batch in enumerate(training_loader):
            loss = model.forward(batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model)
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

        if e % 10 == 0 or e == num_epochs - 1 or patience > max_patience:
            logger.info(f"Epoch {e}: val nll={loss_val:.4f}, patience={patience}")

        if patience > max_patience:
            logger.info(f"Early stopping at epoch {e}.")
            break

    nll_val = np.asarray(nll_val)
    save_training_curve(nll_val, results_path)

    return nll_val, best_model

def reconerrorPIKE(model, data_loader, logger=None):
    """
    Calculate PIKE reconstruction error for all samples in a data loader.
    Args:
        model: Trained VAE model (should have encoder and decoder attributes).
        data_loader: DataLoader for the dataset to evaluate.
        logger: Logger object for logging results (optional).
        log_first_n: Number of individual sample errors to log (default: 10).
    Returns:
        mean_pike: Mean PIKE reconstruction error over all samples.
        pike_errors: List of PIKE errors for all samples.
    """
    pike_errors = []
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, test_batch in enumerate(data_loader):
            for i in range(test_batch.shape[0]):
                x = test_batch[i].cpu().numpy()
                x_tensor = test_batch[i].unsqueeze(0)
                mu, logvar = model.encoder.encode(x_tensor)
                z = mu  # use mean of posterior
                x_hat = model.decoder(z).cpu().numpy().squeeze()
                pike_err = pike_reconstruction_error(x, x_hat)
                pike_errors.append(pike_err)
                total_samples += 1
                if logger is not None and total_samples <= 10:
                    logger.info(f"Test sample {total_samples}: PIKE RE = {pike_err:.4f}")
            if logger is not None:
                logger.info(f"Processed batch {batch_idx+1} of test set.")
    mean_pike = np.mean(pike_errors)
    logger.info(f"Mean PIKE reconstruction error (n={total_samples}): {mean_pike:.4f}")
    return mean_pike, pike_errors


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