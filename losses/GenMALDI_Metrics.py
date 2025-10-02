import numpy as np
from itertools import combinations

from losses.PIKE import calculate_PIKE


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
    Compute unbiased MMD^2 between sets of spectra using PIKE kernel.
    X: list of spectra (each np.ndarray shape (n_peaks, 2))
    Y: list of spectra
    t: bandwidth parameter for PIKE
    Returns: MMD^2 value (float)
    """
    m, n = len(X), len(Y)

    # K_xx
    sum_xx = 0.0
    for i, j in combinations(range(m), 2):
        sum_xx += calculate_PIKE(X[i], X[j], t)
    sum_xx *= 2 / (m * (m - 1))

    # K_yy
    sum_yy = 0.0
    for i, j in combinations(range(n), 2):
        sum_yy += calculate_PIKE(Y[i], Y[j], t)
    sum_yy *= 2 / (n * (n - 1))

    # K_xy
    sum_xy = 0.0
    for i in range(m):
        for j in range(n):
            sum_xy += calculate_PIKE(X[i], Y[j], t)
    sum_xy *= 2 / (m * n)

    return sum_xx + sum_yy - sum_xy


def jaccard_topk(x, y, k=50, mz_tol=0.1):
    """
    Jaccard index between top-k peaks of spectra x and y.
    x, y: np.ndarray of shape (n_peaks, 2) -> [m/z, intensity]
    k: number of top peaks by intensity
    mz_tol: tolerance for considering peaks as matching
    Returns: Jaccard index (float)
    """
    # Select top-k mz values
    x_top = x[np.argsort(x[:, 1])[-k:], 0]
    y_top = y[np.argsort(y[:, 1])[-k:], 0]

    # Match peaks with tolerance
    x_set = set()
    for mz in x_top:
        # round to bin defined by tolerance
        x_set.add(int(mz / mz_tol))

    y_set = set()
    for mz in y_top:
        y_set.add(int(mz / mz_tol))

    inter = len(x_set & y_set)
    union = len(x_set | y_set)

    return inter / union if union > 0 else 0.0


def class_distance(X_gen_class, t=8):
    """
    Average pairwise PIKE distance (1 - PIKE) within a class of generated spectra.
    X_gen_class: list of generated spectra in the same class
    t: bandwidth parameter for PIKE
    Returns: mean pairwise distance
    """
    m = len(X_gen_class)
    if m < 2:
        return 0.0

    sum_dist = 0.0
    count = 0
    for i, j in combinations(range(m), 2):
        sim = calculate_PIKE(X_gen_class[i], X_gen_class[j], t)
        sum_dist += (1 - sim)
        count += 1

    return sum_dist / count


def neighbour_distance(X_gen, Y_train, t=8):
    """
    Nearest-neighbor distance for generated spectra to training set.
    X_gen: list of generated spectra
    Y_train: list of real/training spectra
    Returns: mean nearest-neighbor distance
    """
    dists = []
    for x in X_gen:
        best = -np.inf
        for y in Y_train:
            sim = calculate_PIKE(x, y, t)
            if sim > best:
                best = sim
        dists.append(1 - best)
    return np.mean(dists), dists

