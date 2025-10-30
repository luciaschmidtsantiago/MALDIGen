import os
import torch
import sys
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIKE_GPU import calculate_PIKE_gpu, calculate_pike_matrix
from dataloader.data import load_data, get_dataloaders, compute_mean_spectra_per_label


def get_fold(train_loader, subset_ratio=0.1, seed=42):
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

def compute_pairwise_pike_per_label(subset, device, results_path, t=8):
    """
    Compute and save full pairwise PIKE (subset vs subset) for each label.
    Uses the single-pair kernel to avoid batch-shape surprises.
    Diagonal is set to 1.0; zero spectra are handled safely.
    """
    X_subset, y_subset = subset
    os.makedirs(results_path, exist_ok=True)

    label_ids = torch.unique(y_subset).cpu().tolist()
    print(f"Computing intra-label PIKE matrices for {len(label_ids)} labels")

    with torch.no_grad():
        for label in label_ids:
            label = int(label)
            spectra = X_subset[y_subset == label].to(device)
            n = spectra.shape[0]
            if n < 2:
                print(f"Skipping label {label}: only {n} sample(s)")
                continue

            print(f"\nLabel {label}: computing {n}x{n} PIKE matrix")
            # Ensure 2D [N, D]
            if spectra.ndim == 3 and spectra.shape[1] == 1:
                spectra = spectra.squeeze(1)
            if spectra.ndim == 1:
                spectra = spectra.unsqueeze(0)

            sims = torch.zeros((n, n), dtype=torch.float32, device=device)

            # Precompute zero-vector mask
            zero_mask = (spectra.abs().sum(dim=1) == 0)

            for i in tqdm(range(n), desc=f"label {label} rows", leave=False):
                x_i = spectra[i].view(-1)

                if zero_mask[i]:
                    sims[i, :] = 0.0
                    continue

                for j in range(n):
                    y_j = spectra[j].view(-1)
                    if zero_mask[j]:
                        sims[i, j] = 0.0
                        continue

                    val = calculate_PIKE_gpu(x_i, y_j, t)

                    # --- handle numeric edge cases ---
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0

                    sims[i, j] = float(val)

            # Diagonal = 1.0
            idx = torch.arange(n, device=device)
            sims[idx, idx] = 1.0

            # Save and free
            np.save(os.path.join(results_path, f"pairwise_PIKE_label{label}.npy"),
                    sims.detach().cpu().numpy())
            print(f"✅ Saved PIKE matrix for label {label} ({n}x{n})")

            del sims, spectra
            torch.cuda.empty_cache()

def summarize_intra_label_pike(results_path, label_convergence, y_subset):
    """
    Read stored .npy matrices and write a single-row CSV:
    label, all_<name0>, all_<name1>, ...
    Each cell is mean±std of the off-diagonal entries for that label.
    """
    labels = sorted(torch.unique(y_subset).cpu().tolist())
    label_names = [label_convergence.get(l, str(l)) if isinstance(label_convergence, dict) else str(l)
                   for l in labels]

    csv_path = os.path.join(results_path, "intra_label_pike_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["LABEL"] + [f"{nm}" for nm in label_names])

    row = []
    for l in labels:
        path = os.path.join(results_path, f"pairwise_PIKE_label{int(l)}.npy")
        mat = np.load(path)
        # exclude diagonal (which is 1.0 by design)
        off_diag = mat[~np.eye(mat.shape[0], dtype=bool)]
        mean = float(np.mean(off_diag))
        std = float(np.std(off_diag))
        row.append(f"{mean:.3f}±{std:.3f}")

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(["All_intra_label"] + row)

    print(f"✅ Saved intra-label PIKE summary to {csv_path}")

# ===============================================================
#  METRIC FUNCTIONS (operating on precomputed PIKE matrices)
# ===============================================================
def compute_mmd_from_pike(K_xx, K_yy=None, K_xy=None):
    """
    Compute unbiased MMD^2 using precomputed PIKE kernel matrices.
    If K_yy or K_xy are None, assumes X=Y (same dataset baseline).
    """
    if K_yy is None:
        K_yy = K_xx
    if K_xy is None:
        K_xy = K_xx

    m = K_xx.shape[0]
    n = K_yy.shape[0]
    if m < 2 or n < 2:
        return np.nan

    mask_xx = ~np.eye(m, dtype=bool)
    mask_yy = ~np.eye(n, dtype=bool)

    term_xx = np.mean(K_xx[mask_xx])
    term_yy = np.mean(K_yy[mask_yy])
    term_xy = np.mean(K_xy)

    mmd2 = term_xx + term_yy - 2 * term_xy
    return float(mmd2)

def compute_class_distance_stats(K):
    """Return mean ± std of pairwise distances within a class."""
    m = K.shape[0]
    if m < 2:
        return np.nan, np.nan
    mask = np.triu(np.ones_like(K, dtype=bool), k=1)
    dists = 1.0 - K[mask]
    return float(np.mean(dists)), float(np.std(dists))

def compute_neighbour_distance_stats(K):
    """Return mean ± std of nearest-neighbour distances per sample."""
    m = K.shape[0]
    if m < 2:
        return np.nan, np.nan
    K_ = K.copy()
    np.fill_diagonal(K_, -np.inf)
    nearest = np.max(K_, axis=1)
    dists = 1.0 - nearest
    return float(np.mean(dists)), float(np.std(dists))


# ===============================================================
#  DRIVER FUNCTION
# ===============================================================
def compute_all_baseline_metrics(results_path, label_convergence, y_subset):
    """
    Compute all baseline metrics (PIKE mean±std, MMD², class distance, neighbour distance)
    and save them in a single compact CSV table where each label is a column.
    """

    # -------------------------------------------------------
    # Setup: ensure mapping uses correct species names
    # -------------------------------------------------------
    labels = sorted(torch.unique(y_subset).cpu().tolist())
    label_ids = [int(l) for l in labels]

    # Handle both int and str keys gracefully
    label_names = []
    for l in label_ids:
        if l in label_convergence:
            label_names.append(label_convergence[l])
        elif str(l) in label_convergence:
            label_names.append(label_convergence[str(l)])
        else:
            label_names.append(f"{l}")


    # Prepare containers for each metric row
    row_pike = []
    row_mmd = []
    row_dclass = []
    row_dnn = []

    # -------------------------------------------------------
    # Per-label computation
    # -------------------------------------------------------
    for label_id, label_name in zip(label_ids, label_names):
        path = os.path.join(results_path, f"pairwise_PIKE_label{label_id}.npy")
        if not os.path.exists(path):
            print(f"⚠️ Missing matrix for label {label_id} ({label_name}), skipping.")
            row_pike.append("NaN±NaN")
            row_mmd.append("NaN")
            row_dclass.append("NaN±NaN")
            row_dnn.append("NaN±NaN")
            continue

        K = np.load(path)
        np.fill_diagonal(K, 1.0)
        K = np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)

        # --- PIKE mean±std ---
        off_diag = K[~np.eye(K.shape[0], dtype=bool)]
        mean_pike = float(np.mean(off_diag))
        std_pike = float(np.std(off_diag))
        row_pike.append(f"{mean_pike:.3f}±{std_pike:.3f}")

        # --- Other metrics ---
        mmd2 = compute_mmd_from_pike(K)
        mean_dclass, std_dclass = compute_class_distance_stats(K)
        mean_dnn, std_dnn = compute_neighbour_distance_stats(K)

        print(f"Label {label_name}: MMD²={mmd2:.6f}, D_class={mean_dclass:.6f}±{std_dclass:.6f}, D_NN={mean_dnn:.6f}±{std_dnn:.6f}")

        row_mmd.append(f"{mmd2:.6f}")
        row_dclass.append(f"{mean_dclass:.3f}±{std_dclass:.3f}")
        row_dnn.append(f"{mean_dnn:.3f}±{std_dnn:.3f}")

    # -------------------------------------------------------
    # Write compact CSV table
    # -------------------------------------------------------
    out_csv = os.path.join(results_path, "baseline_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label_ID"] + [str(i) for i in label_ids])
        writer.writerow(["Species_name"] + label_names)
        writer.writerow(["PIKE"] + row_pike)
        writer.writerow(["MMD²"] + row_mmd)
        writer.writerow(["Class_dist"] + row_dclass)
        writer.writerow(["Neighbour_dist"] + row_dnn)

    print(f"✅ Saved compact baseline metrics table to {out_csv}")


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
    X_subset, y_subset = get_fold(train_loader, subset_ratio=0.1, seed=42)

    # GET MEAN SPECTRA PER LABEL (FULL TRAIN)
    # mean_spectra_train, _, _ = compute_mean_spectra_per_label(train_loader, device)

    # COMPUTE PIKE of SUBSET VS MEAN
    # all_pike_per_class = baseline_versus_mean([X_subset, y_subset], mean_spectra_train, device, output_dir)

    # COMPUTE PAIRWISE PIKE PER LABEL (SUBSET VS SUBSET)
    # compute_pairwise_pike_per_label([X_subset, y_subset], device, output_dir, t=8)

    # SUMMARIZE INTRA-LABEL PIKE MATRICES
    # summarize_intra_label_pike(output_dir, label_convergence, y_subset)

    # COMPUTE AND APPEND METRICS
    compute_all_baseline_metrics(output_dir, label_convergence, y_subset)

