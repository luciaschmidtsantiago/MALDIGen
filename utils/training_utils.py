import csv
import os
import sys
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from utils.visualization import save_training_curve
from losses.PIKE import pike_reconstruction_error

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
    nll_train = []
    best_nll = 1000
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        train_loss = 0.
        N_train = 0
        for _, batch in enumerate(training_loader):
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
            logger.info(f"Epoch {e}: train nll={train_loss:.4f}, val nll={loss_val:.4f}, patience={patience}")

        if patience > max_patience:
            logger.info(f"Early stopping at epoch {e}.")
            break

    nll_val = np.asarray(nll_val)
    nll_train = np.asarray(nll_train)
    save_training_curve(nll_train, nll_val, results_path)

    return nll_train, nll_val, best_model


def reconerrorPIKE(model, data_loader, logger=None, labels=None):
    """
    Calculate PIKE reconstruction error for all samples in a data loader, and per class if labels are provided.
    Args:
        model: Trained VAE model (should have encoder and decoder attributes).
        data_loader: DataLoader for the dataset to evaluate.
        logger: Logger object for logging results (optional).
        labels: Optional numpy array or list of labels (same order as data_loader.dataset).
    Returns:
        mean_pike: Mean PIKE reconstruction error over all samples.
        pike_errors: List of PIKE errors for all samples.
        class_pike: Dict mapping class label to average PIKE error (if labels provided, else None).
    """
    from collections import defaultdict
    pike_errors = []
    model.eval()
    class_pike = None
    if labels is not None:
        pike_by_class = defaultdict(list)
        sample_idx = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, test_batch in enumerate(data_loader):
            device = next(model.parameters()).device
            test_batch = test_batch.to(device)
            mu, logvar = model.encoder.encode(test_batch)
            z = mu  # use mean of posterior
            x_hat = model.decoder(z)
            x_true = test_batch.cpu().numpy()
            x_hat = x_hat.cpu().numpy()
            batch_size = x_true.shape[0]
            if labels is not None:
                batch_labels = labels[sample_idx:sample_idx+batch_size]
            # Compute PIKE error for each sample in batch
            for i in range(batch_size):
                pike_err = pike_reconstruction_error(x_true[i], x_hat[i])
                pike_errors.append(pike_err)
                if labels is not None:
                    pike_by_class[batch_labels[i]].append(pike_err)
                if logger is not None and total_samples + i + 1 <= 10:
                    logger.info(f"Test sample {total_samples + i + 1}: PIKE RE = {pike_err:.4f}")
            total_samples += batch_size
            if logger is not None:
                logger.info(f"Processed batch {batch_idx+1} of test set.")
            if labels is not None:
                sample_idx += batch_size
    mean_pike = np.mean(pike_errors)
    if logger is not None:
        logger.info(f"Mean PIKE reconstruction error (n={total_samples}): {mean_pike:.4f}")
    if labels is not None:
        class_pike = {cls: np.mean(errs) for cls, errs in pike_by_class.items()}
        if logger is not None:
            for cls, avg_pike in class_pike.items():
                logger.info(f"Class {cls}: mean PIKE RE = {avg_pike:.4f} (n={len(pike_by_class[cls])})")
    return mean_pike, pike_errors, class_pike

def write_pike_csv(model_path, class_pike_dict, total_pike, csv_path, label_order=None):
    """
    Write PIKE per-class and total PIKE for a model to a CSV file, appending if exists.
    Args:
        model_path: Path to the model file.
        class_pike_dict: Dict mapping class label to PIKE value.
        total_pike: Overall mean PIKE value.
        csv_path: Path to CSV file to write/append.
        label_order: Optional list of labels to fix column order.
    """
    # Prepare header and row
    if label_order is None:
        label_order = sorted(class_pike_dict.keys())
    header = ['model'] + [f'PIKE_{lbl}' for lbl in label_order] + ['PIKE_total']
    row = [model_path] + [class_pike_dict.get(lbl, '') for lbl in label_order] + [total_pike]
    # Write or append
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


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