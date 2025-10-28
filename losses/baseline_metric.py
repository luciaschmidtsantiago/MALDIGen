import os
import torch
import sys
import csv
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIKE_GPU import calculate_PIKE_gpu_batch, calculate_pike_matrix
from dataloader.data import load_data, get_dataloaders, compute_mean_spectra_per_label

def get_fold(train_loader, device, subset_ratio=0.1, seed=42):
    """
    Collect a stratified subset of the training data.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------------------------------
    # 1. Collect all training data (for mean computation and stratified split)
    # --------------------------------------------------------------------------
    print("Collecting all training data...")
    X_all, y_all = [], []
    for x_batch, y_batch in tqdm(train_loader, desc="Collecting batches"):
        if y_batch.ndim > 1:
            y_batch = y_batch.argmax(dim=1)
        X_all.append(x_batch)
        y_all.append(y_batch)
    X_all = torch.cat(X_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    print(f"Total training samples: {len(X_all)}")

    # Stratified subset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_ratio, random_state=seed)
    _, idx_subset = next(sss.split(np.zeros(len(y_all)), y_all.cpu().numpy()))

    X_subset = X_all[idx_subset].to(device)
    y_subset = y_all[idx_subset].to(device)

    # --------------------------------------------------------------------------
    # 2. Stratified 10% subset of the training data
    # --------------------------------------------------------------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_ratio, random_state=seed)
    _, idx_subset = next(sss.split(np.zeros(len(y_all)), y_all.cpu().numpy()))

    X_subset = X_all[idx_subset]
    y_subset = y_all[idx_subset]
    print(f"Subset selected: {len(X_subset)} samples ({subset_ratio*100:.1f}%)")

    return X_subset, y_subset

def baseline_versus_mean(subset, mean_spectra_train, device, results_path):
    """
    Compute PIKE baseline for a stratified subset (10%) of the training set vs. mean spectra per label.
    """

    X_subset, y_subset = subset

    # ----------------------------------------------------------------------
    # 1. Group subset by label → dict[label_id: tensor]
    # ----------------------------------------------------------------------
    subset_spectra_dict = {}
    for label in tqdm(torch.unique(y_subset), desc="Grouping subset by label"):
        label = int(label.item())
        subset_spectra_dict[label] = X_subset[y_subset == label].to(device)

    # ----------------------------------------------------------------------
    # 1. Compute PIKE matrix using the same helper as for generated spectra
    # ----------------------------------------------------------------------
    print("Computing PIKE matrix for training subset...")
    all_pike_per_class = calculate_pike_matrix(
        generated_spectra=subset_spectra_dict,
        mean_spectra_test=mean_spectra_train,
        label_correspondence=label_convergence,
        device=device,
        results_path=results_path,
        saving=True
    )
    print(f"PIKE matrix saved in {results_path}")

    return all_pike_per_class

def baseline_all():
    """
    Compute full pairwise PIKE baseline for all training samples vs all training samples.
    WARNING: very expensive O(N^2) computation and memory!
    """
    pass  # Implementation omitted due to high computational cost


def compute_baseline_metrics(X_train, y_train, label_convergence, output_dir, sample_frac=0.1, t=8, batch_size=64, seed=42):
    """
    Approximate baseline PIKE per label by sampling a stratified subset per label.

    For each label L:
      - let n = number of samples with label L
      - choose k = max(1, ceil(n * sample_frac)) samples (stratified by label implicitly)
      - for each sampled sample s, compute PIKE(s, x) for all x in label L (batched)
      - aggregate mean/std across all comparisons for that label and write CSV

    This reduces compute roughly by factor ~ (sample_frac * 2) vs full pairwise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    assert len(X_train) == len(y_train), "Data and labels length mismatch"

    labels = sorted(np.unique(y_train))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().float().to(device).view(-1)
        return torch.tensor(x, dtype=torch.float32, device=device).view(-1)

    out_csv = os.path.join(output_dir, "baseline_PIKE.csv")
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Mean_PIKE", "Std_PIKE", "N_label", "Sampled_k"]) 

    rng = np.random.default_rng(seed)

    for label in labels:
        # collect all samples for this label
        X_label = [x for x, y in zip(X_train, y_train) if y == label]
        n = len(X_label)
        label_name = label_convergence[label] if isinstance(label_convergence, dict) and label in label_convergence else str(label)
        if n == 0:
            print(f"Skipping label {label_name}: no samples")
            continue

        k = max(1, int(np.ceil(n * sample_frac)))
        # choose k indices without replacement
        idxs = rng.choice(n, size=k, replace=False)

        # convert all to tensors once
        X_label_tensor = [to_tensor(x) for x in X_label]

        all_sims = []
        for idx in tqdm(idxs, desc=f"PIKE approx {label_name}"):
            x_s = X_label_tensor[idx]
            # compare to all samples in batches
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                x2_batch = torch.stack(X_label_tensor[start:end], dim=0)
                x1_batch = x_s.unsqueeze(0).repeat(x2_batch.size(0), 1)
                sims_batch = calculate_PIKE_gpu_batch(x1_batch, x2_batch, t)
                sims_np = sims_batch.detach().cpu().numpy() if isinstance(sims_batch, torch.Tensor) else np.array(sims_batch)
                all_sims.extend(sims_np.tolist())

        if len(all_sims) > 0:
            mean_pike = float(np.mean(all_sims))
            std_pike = float(np.std(all_sims))
        else:
            mean_pike = 0.0
            std_pike = 0.0

        # Save per-label full PIKE values (one value per line) for later analysis
        # sanitize label_name for filename
        safe_label = "".join([c if (c.isalnum() or c in ('-', '_', '.')) else '_' for c in label_name])
        per_label_path = os.path.join(output_dir, f"baseline_PIKE_{safe_label}.csv")
        try:
            if len(all_sims) > 0:
                np.savetxt(per_label_path, np.array(all_sims), fmt="%.6f", header=f"PIKE values for {label_name}")
            else:
                # create empty file with header
                with open(per_label_path, 'w') as pf:
                    pf.write(f"# PIKE values for {label_name}\n")
        except Exception as e:
            print(f"Warning: failed to write per-label PIKE file {per_label_path}: {e}")

        with open(out_csv, "a", newline='') as f:
            csv.writer(f).writerow([label_name, mean_pike, std_pike, n, k])

        print(f"Label {label_name}: N={n}, sampled={k}, PIKE mean={mean_pike:.4f}, std={std_pike:.4f}")

    print(f"Wrote approx PIKE baseline to: {out_csv}")



if __name__ == "__main__":
    output_dir = "results/baselines"
    os.makedirs(output_dir, exist_ok=True)

    pickle_marisma = "pickles/MARISMa_study.pkl"
    pickle_driams = "pickles/DRIAMS_study.pkl"
    get_labels = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test, ood = load_data(pickle_marisma, pickle_driams, logger=None, get_labels=get_labels)
    train_loader, val_loader, test_loader, ood_loader = get_dataloaders(train, val, test, ood, batch_size=64)
    label_convergence = train.label_convergence

    # GET STRATIFIED SUBSET OF TRAINING DATA (10%)
    X_subset, y_subset = get_fold(train_loader, device, subset_ratio=0.1, seed=42)

    # GET MEAN SPECTRA PER LABEL (FULL TRAIN)
    mean_spectra_train, _, _ = compute_mean_spectra_per_label(train_loader, device)

    # COMPUTE PIKE of SUBSET VS MEAN
    all_pike_per_class = baseline_versus_mean([X_subset, y_subset], mean_spectra_train, device, output_dir)

    # X_train = train.data
    # y_train = train.labels
    # compute_baseline_metrics(X_train, y_train, label_convergence, output_dir, sample_frac=0.1, t=8, batch_size=64)