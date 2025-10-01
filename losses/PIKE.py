from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
import numpy as np

class PIKE():
    def __init__(self, t, n_jobs=10):
        self.t = t
        self.n_jobs = n_jobs

    def __call__(self, X_mz, X_i, Y_mz=None, Y_i=None, distance_TH=1e-6):
        if Y_mz is None and Y_i is None:
            Y_mz = X_mz
            Y_i = X_i

        K = np.zeros((X_mz.shape[0], Y_mz.shape[0]))

        # Precompute distances matrix
        positions_x = X_mz[0,:].reshape(-1, 1)
        positions_y = Y_mz[0,:].reshape(-1, 1)
        distances = pairwise_distances(
                positions_x,
                positions_y,
                metric='sqeuclidean'
            )
        distances = np.exp(-distances / (4 * self.t))

        # Estimate how far (in index space) you need to look to find a “significant match” between the spectra.
        d = np.where(distances[0] < distance_TH)[0][0]

        # Parallel computation
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_partial_sum)(i, x, X_i, Y_i, distances, d) 
            for i, x in enumerate(X_i.T)
        )

        K = np.sum(results, axis=0) / (4 * self.t * np.pi)
        return distances, results, K

def compute_partial_sum(i, x, X_i, Y_i, distances, d):
    intensities_y = Y_i.T[:(i+d), :]
    di = distances[i, :(i+d)].reshape(-1, 1)
    prod = intensities_y * di
    x = np.broadcast_to(x, (np.minimum(i+d,  X_i.shape[1]), X_i.shape[0])).T
    return np.matmul(x, prod)
    
def generate_spectrum(array, len):
    """Generates a spectrum.
    Parameters:
    array : list. A list containing two lists: the first with intensities and the second with m/z values.
    len : int. The length of the spectrum.
    Returns:
    mz : np.ndarray. The m/z values of the spectrum.
    spectrum : np.ndarray. The intensity values of the spectrum."""
    peaks_int, peaks_mz = array
    spectrum = np.zeros(len + 1)
    for mz, intensity in zip(peaks_mz, peaks_int):
        if 0 <= mz <= len:
            spectrum[int(mz)] = intensity
    mz = np.linspace(0, len, len + 1)
    return mz, spectrum

def reshape_spectrum(mz, i):
    return mz.reshape(1, -1), i.reshape(1, -1)

def calculate_PIKEtoMean(reconstructed, true, target, t=8):
    """
    Computes normalized PIKE reconstruction error between a generated spectrum and the mean spectrum of each label.
    Args:
        reconstructed: 1D numpy array, reconstructed/synthetic spectrum (intensities, length = mz_max+1).
        true: 2D numpy array, all spectra (n_samples, n_features).
        target: 1D array, labels for each spectrum.
        t: PIKE temperature parameter.
    Returns:
        norm_pike_dict: dict {label: normalized PIKE value}
    """
    species_labels = np.unique(target)
    means = {}
    for label in species_labels:
        idx = np.where(target == label)[0]
        spectra = true[idx]
        means[label] = spectra.mean(axis=0)

    mz_max = reconstructed.shape[0]
    mz_axis = np.arange(mz_max)
    X_mz = mz_axis.reshape(1, -1)
    X_i = reconstructed.reshape(1, -1)
    pike = PIKE(t)
    norm_pike_dict = {}
    _, _, K_gen_gen = pike(X_mz, X_i)
    for label, mean_spectrum in means.items():
        Y_i = mean_spectrum.reshape(1, -1)
        Y_mz = mz_axis.reshape(1, -1)
        _, _, K_gen_mean = pike(X_mz, X_i, Y_mz, Y_i)
        _, _, K_mean_mean = pike(Y_mz, Y_i)
        K_norm = K_gen_mean / np.sqrt(K_gen_gen * K_mean_mean)
        norm_pike_dict[label] = K_norm[0][0]
    return norm_pike_dict

def calculate_PIKE(x, x_hat, t=8):
    """
    Computes the normalized PIKE error between a true spectrum and its reconstruction.
    Args:
        x: 1D numpy array, true/original spectrum (intensities).
        x_hat: 1D numpy array, reconstructed spectrum (intensities).
        t: PIKE temperature parameter.
    Returns:
        norm_pike: float, normalized PIKE error between x and x_hat.
    """
    mz_max = x.shape[0]
    mz_axis = np.arange(mz_max)
    X_mz = mz_axis.reshape(1, -1)
    X_i = x.reshape(1, -1)
    Xhat_i = x_hat.reshape(1, -1)
    pike = PIKE(t)
    _, _, K_x_x = pike(X_mz, X_i)
    _, _, K_xhat_xhat = pike(X_mz, Xhat_i)
    _, _, K_x_xhat = pike(X_mz, X_i, X_mz, Xhat_i)
    norm_pike = K_x_xhat / np.sqrt(K_x_x * K_xhat_xhat)
    return norm_pike[0][0]