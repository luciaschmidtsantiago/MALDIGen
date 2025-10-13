import torch
import numpy as np
from itertools import combinations
import os
import csv

from losses.PIKE_GPU import calculate_PIKE_gpu

# Helper to ensure input is a 1D torch tensor (float32, on device)
def to_tensor_1d(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.float().to(device).view(-1)
    x = np.asarray(x)
    return torch.from_numpy(x.astype(np.float32)).to(device).view(-1)


def pike_kernel(x, y, t=8):
    """
    Compute PIKE kernel between two spectra represented as (mz, intensity) arrays.
    x: np.ndarray of shape (n_peaks, 2) -> [m/z, intensity]
    y: np.ndarray of shape (m_peaks, 2) -> [m/z, intensity]
    t: bandwidth parameter
    """
    mz_x, I_x = x[:, 0], x[:, 1]
    mz_y, I_y = y[:, 0], y[:, 1]

    # Pairwise Gaussian on m/z
    diff = mz_x[:, None] - mz_y[None, :]
    weights = np.exp(- (diff ** 2) / (2 * t ** 2))

    return np.sum((I_x[:, None] * I_y[None, :]) * weights)

def mmd_pike(X, Y, t=8):
    """
    Compute unbiased Maximum Mean Discrepancy (MMD^2) between two sets of spectra using the PIKE kernel.
    X: list of generated spectra (each np.ndarray shape (n_peaks, 2) or (length,))
    Y: list of real spectra
    t: bandwidth parameter for PIKE
    Returns: MMD^2 value (float)
    """
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return np.nan

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = [to_tensor_1d(x, device=device) for x in X]
    Y_t = [to_tensor_1d(y, device=device) for y in Y]

    # K_xx: mean kernel value between all pairs in X (excluding diagonal)
    sum_xx = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                sum_xx += calculate_PIKE_gpu(X_t[i], X_t[j], t)
    K_xx = sum_xx / (m * (m - 1))

    # K_yy: mean kernel value between all pairs in Y (excluding diagonal)
    sum_yy = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                sum_yy += calculate_PIKE_gpu(Y_t[i], Y_t[j], t)
    K_yy = sum_yy / (n * (n - 1))

    # K_xy: mean kernel value between all pairs (X, Y)
    sum_xy = 0.0
    for i in range(m):
        for j in range(n):
            sum_xy += calculate_PIKE_gpu(X_t[i], Y_t[j], t)
    K_xy = sum_xy / (m * n)

    return K_xx + K_yy - 2 * K_xy

# DISCARDED
def jaccard_topk(x, y, k=50, mz_tol=0.1):
    """
    Jaccard index between top-k peaks of spectra x and y.
    x, y: np.ndarray of shape (n_peaks, 2) -> [m/z, intensity]
    k: number of top peaks by intensity
    mz_tol: tolerance for considering peaks as matching
    Returns: Jaccard index (float)
    """
    # Defensive: handle 1D input (intensity only)
    if x.ndim == 1:
        x = np.stack([np.arange(x.shape[0]), x], axis=1)
    if y.ndim == 1:
        y = np.stack([np.arange(y.shape[0]), y], axis=1)
    # Select top-k mz values
    x_top = x[np.argsort(x[:, 1])[-k:], 0]
    y_top = y[np.argsort(y[:, 1])[-k:], 0]
    # Match peaks with tolerance
    x_set = set(int(mz / mz_tol) for mz in x_top)
    y_set = set(int(mz / mz_tol) for mz in y_top)
    inter = len(x_set & y_set)
    union = len(x_set | y_set)
    return inter / union if union > 0 else 0.0


def class_distance(X_gen_class, t=8):
    """
    Average pairwise PIKE distance (1 - PIKE) within a class of generated spectra.
    Returns mean and std of all pairwise distances.
    X_gen_class: list of generated spectra in the same class
    t: bandwidth parameter for PIKE
    Returns: (mean, std) of pairwise distances
    """
    m = len(X_gen_class)
    if m < 2:
        return 0.0, 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = [to_tensor_1d(x, device=device) for x in X_gen_class]
    dists = []
    for i, j in combinations(range(m), 2):
        sim = calculate_PIKE_gpu(X_t[i], X_t[j], t)
        dists.append(1 - sim)
    return np.mean(dists), np.std(dists)


def neighbour_distance(X_gen, Y_train, t=8):
    """
    Nearest-neighbor distance for generated spectra to training set.
    Returns mean and std of nearest-neighbor distances.
    X_gen: list of generated spectra
    Y_train: list of real/training spectra
    Returns: (mean, std) of nearest-neighbor distances
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = [to_tensor_1d(x, device=device) for x in X_gen]
    Y_t = [to_tensor_1d(y, device=device) for y in Y_train]
    dists = []
    for x in X_t:
        best = -np.inf
        for y in Y_t:
            sim = calculate_PIKE_gpu(x, y, t)
            if sim > best:
                best = sim
        dists.append(1 - best)
    return np.mean(dists), np.std(dists)

def save_generative_metrics_csv(generated_spectra, mean_spectra_test, results_path):
    """
    Compute and save mean±std of MMD, Jaccard, class distance, and neighbor distance for each label's generated spectra.
    """
    # Prepare test spectra as list (using means for each label)
    test_spectra_list = [mean_spectra_test[label_id].cpu().numpy().squeeze() for label_id in mean_spectra_test]
    csv_rows = []
    header = ["label", "MMD", "Jaccard_mean±std", "ClassDist_mean±std", "NeighborDist_mean±std"]
    for label_name, gen_specs in generated_spectra.items():
        gen_specs_list = [g for g in gen_specs]
        # MMD (PIKE kernel) for this label (all generated vs all test)
        mmd = mmd_pike(gen_specs_list, test_spectra_list)
        # Jaccard index (average over all pairs)
        jaccard_scores = []
        for g in gen_specs_list:
            for y in test_spectra_list:
                jaccard_scores.append(jaccard_topk(g, y))
        jaccard_mean = np.mean(jaccard_scores)
        jaccard_std = np.std(jaccard_scores)
        # Class distance (within generated for this label)
        class_dist_mean, class_dist_std = class_distance(gen_specs_list)
        # Neighbor distance (generated to test)
        neighbor_dist_mean, neighbor_dist_std = neighbour_distance(gen_specs_list, test_spectra_list)
        csv_rows.append([
            label_name,
            f"{mmd:.4f}",
            f"{jaccard_mean:.4f}±{jaccard_std:.4f}",
            f"{class_dist_mean:.4f}±{class_dist_std:.4f}",
            f"{neighbor_dist_mean:.4f}±{neighbor_dist_std:.4f}"
        ])
    metrics_csv_path = os.path.join(results_path, 'generative_metrics_per_label.csv')
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print(f"Saved generative metrics per label to {metrics_csv_path}")