import torch
import math
import numpy as np
import os
import csv
from tqdm import tqdm

class PIKE_GPU:
    def __init__(self, t=8):
        self.t = t

    def __call__(self, X_mz, X_i, Y_mz=None, Y_i=None, distance_TH=1e-6):
        """
        GPU-accelerated PIKE kernel calculation using PyTorch tensors.
        Args:
            X_mz: (1, N) tensor, m/z positions for X
            X_i: (1, N) tensor, intensities for X
            Y_mz: (1, M) tensor, m/z positions for Y (optional)
            Y_i: (1, M) tensor, intensities for Y (optional)
            distance_TH: threshold for significant match (not used in GPU version)
        Returns:
            distances: (N, M) tensor, kernel distances
            K: scalar, PIKE value
        """
        if Y_mz is None or Y_i is None:
            Y_mz = X_mz
            Y_i = X_i
        device = X_mz.device
        # Compute squared euclidean distances
        positions_x = X_mz[0].unsqueeze(1)  # (N, 1)
        positions_y = Y_mz[0].unsqueeze(1)  # (M, 1)
        distances = (positions_x - positions_y.T) ** 2  # (N, M)
        distances = torch.exp(-distances / (4 * self.t))
        # Broadcast X_i and Y_i for element-wise multiplication
        # X_i: (1, N) -> (N, 1), Y_i: (1, M) -> (1, M)
        X_i_b = X_i.T  # (N, 1)
        Y_i_b = Y_i    # (1, M)
        prod = X_i_b * Y_i_b * distances  # (N, M)
        K = prod.sum() / (4 * self.t * math.pi)
        return distances, K

def generate_spectrum_gpu(array, length, device):
    """Generates a spectrum as torch tensors."""
    peaks_int, peaks_mz = array
    spectrum = torch.zeros(length + 1, device=device)
    for mz, intensity in zip(peaks_mz, peaks_int):
        if 0 <= mz <= length:
            spectrum[int(mz)] = intensity
    mz = torch.linspace(0, length, length + 1, device=device)
    return mz, spectrum

def reshape_spectrum_gpu(mz, i):
    return mz.view(1, -1), i.view(1, -1)

def calculate_PIKE_gpu(x, x_hat, t=8):
    """
    Computes normalized PIKE error between true and reconstructed spectrum using GPU.
    Args:
        x: 1D torch tensor, true spectrum (intensities)
        x_hat: 1D torch tensor, reconstructed spectrum (intensities)
        t: PIKE temperature parameter
    Returns:
        norm_pike: float, normalized PIKE error
    """
    device = x.device
    mz_max = x.shape[0] - 1
    mz_axis = torch.arange(mz_max + 1, device=device)
    X_mz, X_i = mz_axis.view(1, -1), x.view(1, -1)
    Xhat_i = x_hat.view(1, -1)
    pike = PIKE_GPU(t)
    _, K_x_x = pike(X_mz, X_i)
    _, K_xhat_xhat = pike(X_mz, Xhat_i)
    _, K_x_xhat = pike(X_mz, X_i, X_mz, Xhat_i)
    norm_pike = K_x_xhat / torch.sqrt(K_x_x * K_xhat_xhat)
    return norm_pike.item()

def calculate_PIKE_gpu_batch(x1_batch, x2_batch, t):
    """
    Batchwise PIKE calculation for batches of pairs.
    Args:
        x1_batch: Tensor of shape (batch_size, feature_dim)
        x2_batch: Tensor of shape (batch_size, feature_dim)
        t: PIKE parameter
    Returns:
        Tensor of PIKE values, shape (batch_size,)
    """
    # Use PIKE_GPU class for each pair, matching calculate_PIKE_gpu logic
    results = []
    pike_kernel = PIKE_GPU(t)
    device = x1_batch.device
    mz_max = x1_batch.shape[1] - 1
    mz_axis = torch.arange(mz_max + 1, device=device)
    X_mz = mz_axis.view(1, -1)
    for x1, x2 in zip(x1_batch, x2_batch):
        X_i = x1.view(1, -1)
        Xhat_i = x2.view(1, -1)
        _, K_x_x = pike_kernel(X_mz, X_i)
        _, K_xhat_xhat = pike_kernel(X_mz, Xhat_i)
        _, K_x_xhat = pike_kernel(X_mz, X_i, X_mz, Xhat_i)
        norm_pike = K_x_xhat / torch.sqrt(K_x_x * K_xhat_xhat)
        results.append(norm_pike.item())
    return torch.tensor(results, device=device)

def calculate_pike_matrix(generated_spectra, mean_spectra_test, label_correspondence, device, results_path=None, saving=True):
    """
    Calculate PIKE to all class means for each generated spectrum in generated_spectra.
    Args:
        generated_spectra: dict {label_name: np.ndarray or tensor of generated spectra}
        mean_spectra_test: dict {label_name: mean spectrum tensor}
        device: torch.device
        results_path: path to save CSV
        saving: whether to save CSV
    Returns: all_pike_per_class
    """
    all_pike_per_class = {}
    with torch.no_grad():
        for label_name in tqdm(generated_spectra, desc="Labels"):
            spectra = generated_spectra[label_name]
            # spectra: [n_generate, D] (np.ndarray or tensor)
            spectra = spectra.to(device)
            mean_spectra_all = {lab: mean_spectra_test[lab].to(device).squeeze() for lab in mean_spectra_test}
            pike_per_class = []
            for i in range(spectra.shape[0]):
                x_i_tensor = spectra[i]
                pike_dict = {}
                for other_lab, mean_spec in mean_spectra_all.items():
                    pike_to_other = calculate_PIKE_gpu(x_i_tensor, mean_spec)
                    pike_dict[other_lab] = pike_to_other
                pike_per_class.append(pike_dict)
            for idx, pike_dict in enumerate(pike_per_class):
                for other_lab, val in pike_dict.items():
                    key = (label_name, other_lab)
                    if key not in all_pike_per_class:
                        all_pike_per_class[key] = []
                    all_pike_per_class[key].append(val)
    # Save matrix if requested
    if saving and results_path is not None:
        row_labels = sorted(set(gen for (gen, _) in all_pike_per_class.keys()))
        col_labels = sorted(set(mean for (_, mean) in all_pike_per_class.keys()))
        label_id_to_name = {str(k): v for k, v in label_correspondence.items()} if isinstance(label_correspondence, dict) else None
        def col_label_str(lab):
            if label_id_to_name and str(lab) in label_id_to_name:
                return f"mean_{label_id_to_name[str(lab)]}"
            else:
                return f"mean_{lab}"
        matrix = []
        for gen_label in row_labels:
            row = []
            for mean_label in col_labels:
                key = (gen_label, mean_label)
                if key in all_pike_per_class:
                    values_np = np.array([v.item() if hasattr(v, 'item') else float(v) for v in all_pike_per_class[key]])
                    mean = np.mean(values_np)
                    std = np.std(values_np)
                    cell = f"{mean:.3f}±{std:.3f}"
                else:
                    cell = ""
                row.append(cell)
            matrix.append(row)
        pike_csv_path = os.path.join(results_path, 'all_labels_pike_matrix.csv')
        with open(pike_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["label"] + [col_label_str(lab) for lab in col_labels])
            for i, gen_label in enumerate(row_labels):
                writer.writerow([gen_label] + matrix[i])
        print(f"Saved PIKE matrix to {pike_csv_path}")
    return all_pike_per_class