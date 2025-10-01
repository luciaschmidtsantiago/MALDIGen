# data/data.py
import numpy as np
import torch
import pickle
import logging

from torch.utils.data import Dataset, DataLoader

class MALDI(Dataset):
    """
    PyTorch Dataset for MALDI-TOF spectra.
    Each sample is one spectrum, optionally normalized to [0,1].
    """
    def __init__(self, data, labels, normalization=True, get_labels=False):
        self.data = data
        self.labels = labels
        self.normalization = normalization
        self.get_labels = get_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrum = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.normalization:
            min_val, max_val = spectrum.min(), spectrum.max()
            spectrum = (spectrum - min_val) / (max_val - min_val + 1e-8)
        if self.get_labels:
            label = self.labels[idx]
            return spectrum, label
        else:
            return spectrum

def load_data(pickle_marisma, pickle_driams, logger: logging.Logger):
    """
    Load MARISMa and DRIAMS pickled datasets, split into
    train/val/test/OOD, and wrap them in MALDI Dataset objects.
    """
    logger.info("=" * 80)
    logger.info("DATA CONFIGURATION")
    logger.info("=" * 80)

    # --- MARISMa ---
    with open(pickle_marisma, "rb") as f:
        data_marisma = pickle.load(f)
    spectra_m, labels_m, metas_m = data_marisma['data'], data_marisma['label'], data_marisma['meta']
    years = np.array([m['year'] for m in metas_m])

    train_idx = np.where((years >= 2019) & (years <= 2023))[0]
    val_idx   = np.where(years == 2018)[0]
    test_idx  = np.where(years == 2024)[0]
    
    # --- DRIAMS ---
    with open(pickle_driams, "rb") as f:
        data_driams = pickle.load(f)
    spectra_d, labels_d, metas_d = data_driams['data'], data_driams['label'], data_driams['meta']
    hospitals = np.array([m['hospital'] for m in metas_d])

    train_idx_d = np.where((hospitals == 'DRIAMS_A') | (hospitals == 'DRIAMS_B'))[0]
    ood_idx_d   = np.where((hospitals == 'DRIAMS_C') | (hospitals == 'DRIAMS_D'))[0]

    # --- Merge MARISMa train + DRIAMS A/B train ---
    spectra_train = np.concatenate([spectra_m[train_idx], spectra_d[train_idx_d]])
    labels_train  = np.concatenate([labels_m[train_idx], labels_d[train_idx_d]])

    # Create Dataset objects
    train = MALDI(spectra_train, labels_train)
    val_m   = MALDI(spectra_m[val_idx], labels_m[val_idx])
    test_m  = MALDI(spectra_m[test_idx], labels_m[test_idx])
    ood_d   = MALDI(spectra_d[ood_idx_d], labels_d[ood_idx_d])

    logger.info("Training with MARISMa (2018-2023) and DRIAMS (A, B)")
    logger.info(f"----Training set size: {len(train)}")
    logger.info("Validation with MARISMa (2018)")
    logger.info(f"Validation set size: {len(val_m)}")
    logger.info("Testing with MARISMa (2024)")
    logger.info(f"----Test set size: {len(test_m)}")
    logger.info("OOD testing with DRIAMS (C, D)")
    logger.info(f"----OOD set size: {len(ood_d)}")
    logger.info(f"Labels: {np.unique(labels_train)}")

    return train, val_m, test_m, ood_d

def get_dataloaders(train_ds, val_ds, test_ds, ood_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)
    ood_loader   = DataLoader(ood_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader, ood_loader