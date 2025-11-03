# dataloader/data.py
import numpy as np
import torch
import pickle
import json
import os
from torch.utils.data import Dataset, DataLoader

class MALDI(Dataset):
    """
    PyTorch Dataset for MALDI-TOF spectra.
    Each sample is one spectrum, optionally normalized to [0,1].
    """
    def __init__(self, data, labels, normalization=True, get_labels=False):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.normalization = normalization
        self.get_labels = get_labels

        if isinstance(labels[0], str):
            label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
            labels = np.vectorize(label_map.get)(labels)
            self.label_convergence = {v: str(k) for k, v in label_map.items()}  # int → string
        else:
            self.label_convergence = None

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.n_classes = int(self.labels.max().item()) + 1 if len(self.labels) > 0 else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrum = self.data[idx]
        label = self.labels[idx]

        if self.normalization:
            min_val, max_val = spectrum.min(), spectrum.max()
            spectrum = (spectrum - min_val) / (max_val - min_val + 1e-8)
           
            # DIFFUSION MODEL NORMALIZATION
            if self.normalization == 'diffusion':
                spectrum = spectrum * 2.0 - 1.0
                if spectrum.dim() == 1:
                    spectrum = spectrum.unsqueeze(0)
                label = label.long()
                # convert to one-hot float32 and place one-hot on same device as spectrum
                one_hot = torch.zeros(self.n_classes, dtype=torch.float32, device=spectrum.device)
                one_hot[label] = 1.0
                return spectrum, one_hot
        
        if self.get_labels:
            return spectrum, label
        else:
            return spectrum

def compute_and_save_statistics(
    spectra_m, labels_m, metas_m,
    spectra_d, labels_d, metas_d,
    train_idx, val_idx, test_idx, train_idx_d, ood_idx_d,
    stats_path, logger=None
):
    """
    Compute and save statistics about the number of spectra per set and per label/source.
    """
    stats = {}

    # Helper to count per label/species
    def count_labels(labels, metas, idxs, label_key='species', extra_keys=None):
        counts = {}
        for i in idxs:
            meta = metas[i]
            label = meta.get(label_key, str(labels[i]))
            # Compose extra info (e.g., year, hospital)
            extra = {}
            if extra_keys:
                for k in extra_keys:
                    extra[k] = meta.get(k, None)
            if label not in counts:
                counts[label] = {'count': 0, 'extra': {}}
            counts[label]['count'] += 1
            # Count by extra keys (e.g., year, hospital)
            for k, v in extra.items():
                if v is not None:
                    if k not in counts[label]['extra']:
                        counts[label]['extra'][v] = 0
                    counts[label]['extra'][v] = counts[label]['extra'].get(v, 0) + 1
        return counts


    # --- Total training statistics (MARISMa train + DRIAMS A/B train) ---
    total_train_idxs = list(train_idx) + list(train_idx_d)
    # For total, combine metas and labels from both sources
    total_labels = list(labels_m[i] for i in train_idx) + list(labels_d[i] for i in train_idx_d)
    total_metas = list(metas_m[i] for i in train_idx) + list(metas_d[i] for i in train_idx_d)
    stats['Total_train'] = {
        'total': len(total_train_idxs),
        'labels': count_labels(total_labels, total_metas, range(len(total_train_idxs)), label_key='species')
    }

    # MARISMa train (years 2018-2022)
    marisma_train_years = [metas_m[i]['year'] for i in train_idx]
    stats['MARISMa_train'] = {
        'years': sorted(list(set(marisma_train_years))),
        'total': len(train_idx),
        'labels': count_labels(labels_m, metas_m, train_idx, label_key='species', extra_keys=['year'])
    }

    # DRIAMS train (A)
    d_a_idx = [i for i in train_idx_d if metas_d[i]['hospital'] == 'DRIAMS_A']
    stats['DRIAMS_A_train'] = {
        'hospital': 'DRIAMS_A',
        'total': len(d_a_idx),
        'labels': count_labels(labels_d, metas_d, d_a_idx, label_key='species')
    }

    # DRIAMS train (B)
    d_b_idx = [i for i in train_idx_d if metas_d[i]['hospital'] == 'DRIAMS_B']
    stats['DRIAMS_B_train'] = {
        'hospital': 'DRIAMS_B',
        'total': len(d_b_idx),
        'labels': count_labels(labels_d, metas_d, d_b_idx, label_key='species')
    }

    # MARISMa val (2023)
    marisma_val_years = [metas_m[i]['year'] for i in val_idx]
    stats['MARISMa_val'] = {
        'years': sorted(list(set(marisma_val_years))),
        'total': len(val_idx),
        'labels': count_labels(labels_m, metas_m, val_idx, label_key='species', extra_keys=['year'])
    }

    # MARISMa test (2024)
    marisma_test_years = [metas_m[i]['year'] for i in test_idx]
    stats['MARISMa_test'] = {
        'years': sorted(list(set(marisma_test_years))),
        'total': len(test_idx),
        'labels': count_labels(labels_m, metas_m, test_idx, label_key='species', extra_keys=['year'])
    }

    # DRIAMS OOD (C and D)
    ood_idx_d = [i for i in ood_idx_d if metas_d[i]['hospital'] in ['DRIAMS_C', 'DRIAMS_D']]
    stats['DRIAMS_ood'] = {
        'hospitals': ['DRIAMS_C', 'DRIAMS_D'],
        'total': len(ood_idx_d),
        'labels': count_labels(labels_d, metas_d, ood_idx_d, label_key='species', extra_keys=['hospital'])
    }

    # DRIAMS OOD (C)
    d_c_idx = [i for i in ood_idx_d if metas_d[i]['hospital'] == 'DRIAMS_C']
    stats['DRIAMS_C_ood'] = {
        'hospital': 'DRIAMS_C',
        'total': len(d_c_idx),
        'labels': count_labels(labels_d, metas_d, d_c_idx, label_key='species')
    }

    # DRIAMS OOD (D)
    d_d_idx = [i for i in ood_idx_d if metas_d[i]['hospital'] == 'DRIAMS_D']
    stats['DRIAMS_D_ood'] = {
        'hospital': 'DRIAMS_D',
        'total': len(d_d_idx),
        'labels': count_labels(labels_d, metas_d, d_d_idx, label_key='species')
    }

    # Save to JSON
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    if logger:
        logger.info(f"Saved MALDI dataset statistics to {stats_path}")
    else:
        print(f"Saved MALDI dataset statistics to {stats_path}")
    return stats

def load_data(pickle_marisma, pickle_driams, logger=None, get_labels=False, model_type=None):
    """
    Load MARISMa and DRIAMS pickled datasets, split into
    train/val/test/OOD, and wrap them in MALDI Dataset objects.
    """
    if logger is not None:
        logger.info("=" * 80)
        logger.info("DATA CONFIGURATION")
        logger.info("=" * 80)

    # --- MARISMa ---
    with open(pickle_marisma, "rb") as f:
        data_marisma = pickle.load(f)
    spectra_m, labels_m, metas_m = data_marisma['data'], data_marisma['label'], data_marisma['meta']
    years = np.array([int(m['year']) for m in metas_m])

    train_idx = np.where(years <= 2022)[0]
    val_idx   = np.where(years == 2023)[0]
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
    normalization = 'diffusion' if model_type == 'diffusion' else True
    train = MALDI(spectra_train, labels_train, normalization=normalization, get_labels=get_labels)
    val_m   = MALDI(spectra_m[val_idx], labels_m[val_idx], normalization=normalization, get_labels=get_labels)
    test_m  = MALDI(spectra_m[test_idx], labels_m[test_idx], normalization=normalization, get_labels=get_labels)
    ood_d   = MALDI(spectra_d[ood_idx_d], labels_d[ood_idx_d], normalization=normalization, get_labels=get_labels)

    # Compute and save statistics
    stats_dir = os.path.join(os.path.dirname(pickle_marisma), '..', 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, 'maldi_statistics.json')
    compute_and_save_statistics(
        spectra_m, labels_m, metas_m,
        spectra_d, labels_d, metas_d,
        train_idx, val_idx, test_idx, train_idx_d, ood_idx_d,
        stats_path, logger=logger
    )

    return train, val_m, test_m, ood_d

def get_dataloaders(train_ds, val_ds, test_ds, ood_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)
    ood_loader   = DataLoader(ood_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader, ood_loader

def compute_summary_spectra_per_label(loader, device=None, logger=None):
    """
    Compute the mean, std, max, and min spectrum per label from a DataLoader.
    Returns a dictionary mapping label -> (mean_spectrum, std_spectrum, max_spectrum, min_spectrum).
    Args:
        loader: DataLoader providing (spectrum, label) batches.
        device: torch.device to perform computations on. If None, uses cuda if available.
        logger: Optional logger for logging information.
    Returns:
        summary_spectra: dict mapping label -> (mean_spectrum, std_spectrum, max_spectrum, min_spectrum).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_x, all_y = [], []

    for batch in loader:
        # Support batches with more than two elements (e.g., (x, y, ...))
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch
        all_x.append(x)
        all_y.append(y)

    X = torch.cat(all_x, dim=0).to(device)
    y = torch.cat(all_y, dim=0).to(device)

    unique_labels = torch.unique(y)
    summary_spectra = {}

    for label in unique_labels:
        mask = (y == label)

        # Compute mean and std spectra
        mean_spec = X[mask].mean(dim=0, keepdim=True)
        std_spec = X[mask].std(dim=0, keepdim=True)

        # Compute global maximum and minimum spectra (per-feature)
        maximum = X[mask].max(dim=0, keepdim=True)[0]
        minimum = X[mask].min(dim=0, keepdim=True)[0]

        summary_spectra[int(label.item())] = (mean_spec, std_spec, maximum, minimum)

        msg = f"Label {int(label.item())}: {mask.sum().item()} samples, mean/std spectrum shape {mean_spec.shape}"
        logger.info(msg) if logger else print(msg)

    return summary_spectra