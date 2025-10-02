import torch
import math

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
