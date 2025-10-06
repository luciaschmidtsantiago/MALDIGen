import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from scipy import interpolate


def spectra_comparison(spectra_list, title="Spectra Comparison"):
    """
    Plots multiple spectra on the same graph for visual comparison.

    Parameters:
    - spectra_list (list of tuples): Each tuple should be (mz, intensity, metadata_label).
    - title (str, optional): Title of the plot.
    
    Example of `spectra_list` input:
        [(mz1, intensity1, "Escherichia-Coli-DRIAMS_A-2015"),
         (mz2, intensity2, "Klebsiella-Pneumoniae-DRIAMS_B-2016")]
    """
    if len(spectra_list) == 0:
        print("⚠️ No spectra to compare.")
        return

    plt.figure()

    # Generate random colors for each spectrum
    random.seed(42)  # Ensure consistent colors across runs
    colors = [plt.cm.viridis(i / len(spectra_list)) for i in range(len(spectra_list))]

    # Plot each spectrum
    for i, (spectrum, label) in enumerate(spectra_list):
        mz, intensity = spectrum.mz, spectrum.intensity
        plt.plot(mz, intensity, label=label, color=colors[i], linewidth=1.5)

    # Formatting
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)  # Show legend with metadata labels
    plt.grid(alpha=0.3)
    plt.show()

def interpolate_spectrum(spectrum, target_length=1000):
    """Interpolates a spectrum to a fixed length."""
    if len(spectrum[0]) < 2:  # Ensure spectrum has at least two points
        raise ValueError("Spectrum must have at least two points for interpolation.")

    # Create interpolation function
    f = interpolate.interp1d(spectrum[0], spectrum[1], kind='linear', fill_value="extrapolate")

    # Generate new m/z values
    mz_new = np.linspace(spectrum[0].min(), spectrum[0].max(), target_length)
    intensity_new = f(mz_new)

    return intensity_new  # Only return interpolated intensities

def plot_pca(dataset, target="bacteria", target_length=1000, n_components=2, title = None):
    """
    Performs PCA on multiple datasets and plots the first two principal components.

    Parameters:
    - dataset (DRIAMS_Dataset): Dataset to visualize.
    - target (str, optional): What to group spectra by (options: "bacteria", "hospital", "year").
    - target_length (int, optional): Fixed length for interpolation before PCA.
    - n_components (int, optional): Number of PCA components to use.

    Returns:
    - None (displays a scatter plot)
    """

    title = f"PCA Projection of Spectra by {target.capitalize()}" if title is None else title

    X, y = [], []

    for i in range(len(dataset)):
        spectrum, metadata = dataset[i]  # Retrieve spectrum data
        mz, intensity = spectrum.mz, spectrum.intensity
        interpolated_intensity = interpolate_spectrum((mz, intensity), target_length)

        # Skip NaN-containing spectra
        if np.isnan(interpolated_intensity).any():
            continue

        X.append(interpolated_intensity)

        # Extract label based on the target grouping (bacteria, hospital, or year)
        genus, species, hospital, year = metadata.split("-")
        if target == "bacteria":
            y.append(f"{genus} {species}")
        elif target == "hospital":
            y.append(hospital)
        elif target == "year":
            y.append(year)
        else:
            raise ValueError("Invalid target. Choose 'bacteria', 'hospital', or 'year'.")

    X = np.array(X)

    if len(X) == 0:
        raise ValueError("No valid spectra available for PCA after removing NaN values.")

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Create color map
    unique_labels = list(set(y))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # Plot PCA result
    plt.figure(figsize=(6, 4))
    for i, label in enumerate(unique_labels):
        mask = np.array(y) == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, color=colors[i], label=label)

    # Labels and Title
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show plot
    plt.show()

def plot_tsne(dataset, target="bacteria", target_length=1000, perplexity=30, learning_rate=200, title=None):
    """
    Plots a t-SNE visualization for one or multiple datasets in a single plot.

    Parameters:
    - datasets (DRIAMS_Dataset): Dataset to visualize.
    - target (str, optional): What to group spectra by (options: "bacteria", "hospital", "year").
    - target_length (int, optional): Fixed length for interpolation before t-SNE.
    - perplexity (int, optional): Perplexity parameter for t-SNE.
    - learning_rate (int, optional): Learning rate parameter for t-SNE.
    - title (str, optional): Custom title for the plot (default is auto-generated).

    Returns:
    - None (displays a scatter plot)
    """

    # Auto-generate title if none provided
    if title is None:
        title = f"t-SNE Projection of Spectra by {target.capitalize()}"

    X, y = [], []


    for i in range(len(dataset)):
        spectrum, metadata = dataset[i]  # Retrieve spectrum data
        mz, intensity = spectrum.mz, spectrum.intensity
        interpolated_intensity = interpolate_spectrum((mz, intensity), target_length)

        # Skip NaN-containing spectra
        if np.isnan(interpolated_intensity).any():
            continue

        X.append(interpolated_intensity)

        # Extract label based on the target grouping
        genus, species, hospital, year = metadata.split("-")
        if target == "bacteria":
            y.append(f"{genus} {species}")  # Example: "Escherichia Coli"
        elif target == "hospital":
            y.append(hospital)  # Example: "DRIAMS_A"
        elif target == "year":
            y.append(year)  # Example: "2018"
        else:
            raise ValueError("Invalid target. Choose 'bacteria', 'hospital', or 'year'.")

    X = np.array(X)

    if len(X) == 0:
        raise ValueError("No valid spectra available for t-SNE after removing NaN values.")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)

    # Create color map
    unique_labels = list(set(y))
    color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # Plot t-SNE result
    plt.figure(figsize=(6, 4))
    for i, label in enumerate(unique_labels):
        mask = np.array(y) == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.6, color=color_map[i], label=label)

    # Labels and Title
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show plot
    plt.show()

def visualize_preprocessing_steps(spectrum, pipeline):
    """
    Visualizes a random spectrum at each step of the preprocessing pipeline.

    Parameters:
    - spectrum (SpectrumObject): The original spectrum.
    - pipeline (SequentialPreprocessor): The preprocessing pipeline.
    
    Returns:
    - None (Displays plots)
    """
    plt.figure(figsize=(10, 6))

    # Start with original spectrum
    spectrum, metadata = spectrum
    mz, intensity = spectrum.mz, spectrum.intensity
    plt.plot(mz, intensity, label=metadata, linestyle="dashed", alpha=0.8)

    # Apply preprocessing step by step
    for step in pipeline.preprocessors:
        spectrum = step(spectrum)  # Apply step
        mz, intensity = spectrum.mz, spectrum.intensity  # Extract new values
        plt.plot(mz, intensity, label=f"After {step.__class__.__name__}")

    # Formatting
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title("Preprocessing Step-by-Step on a Random Sample")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

#### MLP_VAE ####

def plot_tsne(z, data, n=5, path='.', name='vae', perplexity=30, random_state=42):

    effective_perplexity = min(perplexity, max(5, len(z) - 1))

    tsne_z = TSNE(n_components=2, perplexity=effective_perplexity, random_state=random_state)
    z_2d = tsne_z.fit_transform(z)

    if isinstance(n, int):
        np.random.seed(random_state)
        highlight_indices = np.random.choice(len(data), size=n, replace=False)
    else:
        highlight_indices = np.array(n)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.4, label="All samples")
    plt.scatter(z_2d[highlight_indices, 0], z_2d[highlight_indices, 1], color='red', s=20, label="Highlighted samples")

    for i in highlight_indices:
        plt.annotate(str(i), (z_2d[i, 0], z_2d[i, 1]), fontsize=6, alpha=0.7)

    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.title("t-SNE of VAE Latent z")
    plt.legend() 
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{name}_tsne.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ t-SNE plot saved to {filename}")

def plot_pca_2d(z, data, n=5, path='.', name='vae'):
    if isinstance(n, int):
        np.random.seed(42)
        highlight_indices = np.random.choice(len(data), size=n, replace=False)
    else:
        highlight_indices = np.array(n)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.4, label="All samples")
    plt.scatter(z_2d[highlight_indices, 0], z_2d[highlight_indices, 1], color='red', s=20, label="Highlighted samples")

    for i in highlight_indices:
        plt.annotate(str(i), (z_2d[i, 0], z_2d[i, 1]), fontsize=6, alpha=0.8)

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("2D PCA of VAE Latent z")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{name}_pca2d.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ 2D PCA plot saved to {filename}")

def plot_pca_3d(z, data, n=5, path='.', name='vae'):
    if isinstance(n, int):
        np.random.seed(42)
        highlight_indices = np.random.choice(len(data), size=n, replace=False)
    else:
        highlight_indices = np.array(n)

    pca = PCA(n_components=3)
    z_3d = pca.fit_transform(z)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], s=10, alpha=0.4, label="All samples")
    ax.scatter(z_3d[highlight_indices, 0], z_3d[highlight_indices, 1], z_3d[highlight_indices, 2], color='red', s=20, label="Highlighted samples")

    for i in highlight_indices:
        ax.text(z_3d[i, 0], z_3d[i, 1], z_3d[i, 2], str(i), size=6, zorder=1)

    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_zlabel("PCA-3")
    plt.title("3D PCA of VAE Latent z")
    plt.legend()
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{name}_pca3d.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ 3D PCA plot saved to {filename}")

def visualize_preprocessing(sample, pipeline, path, histogram=True):
    """
    Visualizes a spectrum at each step of the preprocessing pipeline with optional histograms.

    Parameters:
    - spectrum (SpectrumObject): The original spectrum.
    - pipeline (SequentialPreprocessor): The preprocessing pipeline.
    - histogram (bool): Whether to plot histograms of intensities after each step.
    """
    spectrum, metadata = sample
    steps = [("Raw Spectrum", spectrum)]

    # Apply preprocessing steps and collect results
    for step in pipeline.preprocessors:
        spectrum = step(spectrum)
        processing_step = step.__class__.__name__
        if processing_step == "Binner":
            processing_step = f'{processing_step} (bin_size={step.step})'
        elif processing_step == "LogScaler":
            processing_step = f'{processing_step} (base={step.base})'
        steps.append((processing_step, spectrum)) 

    n_steps = len(steps)
    ncols = 2 if histogram else 1
    figsize = (14, 3.5 * n_steps) if histogram else (8, 3.5 * n_steps)

    fig, axes = plt.subplots(n_steps, ncols, figsize=figsize, squeeze=False)

    for i, (title, spec) in enumerate(steps):
        mz = spec.mz
        intensity = spec.intensity

        # Plot spectrum
        axes[i, 0].plot(mz, intensity, linewidth=1.2)
        axes[i, 0].set_title(f"{title}")
        axes[i, 0].set_xlabel("m/z")
        axes[i, 0].set_ylabel("Intensity")
        axes[i, 0].grid(alpha=0.3)

        # Plot histogram of intensities (non-zero only)
        if histogram:
            nonzero = intensity[intensity > 0]
            zero_count = np.sum(intensity == 0)

            axes[i, 1].hist(nonzero, bins=50, color='tab:blue', alpha=0.8)
            axes[i, 1].set_title(f"Histogram after {title} (zeros: {zero_count})")
            axes[i, 1].set_xlabel("Intensity (non-zero)")
            axes[i, 1].set_ylabel("Frequency")
            axes[i, 1].grid(alpha=0.3)

    plt.tight_layout()

    metadata = metadata.replace("/", "_")
    filename = os.path.join(path, f"preproc_{metadata}.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Preprocessing plot saved to {filename}")

def plot_samples(samples, path, name="reconstruction", labels=None):
    """
    Plots and saves comparisons between original and reconstructed spectra for multiple samples.

    Parameters:
    - samples (list of tuples): Each tuple is (original_tensor, reconstructed_tensor, index).
    - path (str): Directory to save the plot.
    - name (str): Base filename.
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples), sharex=True)

    # If only one sample, axes is not iterable
    if num_samples == 1:
        axes = [axes]

    for ax, (original, reconstructed, sample_idx) in zip(axes, samples):
        ax.plot(original, label="Original")
        ax.plot(reconstructed, label="Reconstructed", alpha=0.6)
        label_str = f" - {labels[sample_idx]}" if labels and sample_idx in labels else ""
        ax.set_title(f"Sample {sample_idx}: {label_str}")
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f"{name}_samples.pdf"), bbox_inches='tight')
    plt.close()
    print(f"✅ Synthetic samples plot saved to {os.path.join(path, name + '_samples.pdf')}")

def get_mean_spectra(spectra, labels, path, name):
    assert len(spectra) == len(labels), "Number of spectra and labels must match."
    plt.figure(figsize=(10, 6))

    for i, spectra_set in enumerate(spectra):
        mean_spectrum = np.mean(spectra_set, axis=0)

        alpha = 0.6 if i == 1 else 1.0
        plt.plot(mean_spectrum, label=labels[i], color=f"C{i}", alpha=alpha)
        plt.title(f"Mean Spectrum")
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.legend()

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f"{name}_mean_spectra.pdf"), bbox_inches='tight')
    plt.close()
    print(f"✅ Mean spectra plot saved to {os.path.join(path, name + '_mean_spectra.pdf')}")

def plot_umap(z, data, n=5, path='.', name='vae', n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Plots a UMAP projection of latent space `z`.

    Parameters:
    - z (np.array): Latent space array.
    - data (list): List used to determine highlight samples.
    - n (int or list): If int, number of random samples to highlight. If list, specific indices to highlight.
    - path (str): Directory to save the figure.
    - name (str): Base name for the file.
    - n_neighbors (int): UMAP's number of neighbors.
    - min_dist (float): UMAP's minimum distance.
    - random_state (int): Seed for reproducibility.
    """
    
    if UMAP is None:
        print("❌ UMAP is not available. Please install umap-learn: pip install umap-learn")
        return

    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    z_umap = umap_model.fit_transform(z)

    if isinstance(n, int):
        np.random.seed(random_state)
        highlight_indices = np.random.choice(len(data), size=n, replace=False)
    else:
        highlight_indices = np.array(n)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], s=10, alpha=0.4, label="All samples")
    plt.scatter(z_umap[highlight_indices, 0], z_umap[highlight_indices, 1], color='red', s=20, label="Highlighted samples")

    for i in highlight_indices:
        plt.annotate(str(i), (z_umap[i, 0], z_umap[i, 1]), fontsize=6, alpha=0.8)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP of VAE Latent z")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{name}_umap.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ UMAP plot saved to {filename}")