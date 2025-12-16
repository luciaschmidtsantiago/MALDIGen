import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from losses.PIKE_GPU import calculate_PIKE_gpu
from visualization.plotting_utils import LABEL_TO_HEX, printed_names

def safe_load_array(path, device="cpu"):
    """
    Robust loader for arrays saved in several formats.

    Attempts (in order):
      1. numpy.load(..., allow_pickle=False) for plain arrays
      2. numpy.load(..., allow_pickle=True) to handle object arrays and
         pickled dicts/lists containing tensors/arrays
      3. torch.load to support files saved with PyTorch

    Returns a contiguous numpy.ndarray when possible or raises a
    RuntimeError if the file cannot be parsed.
    """
    # Try plain numpy load first (fastest, safest)
    try:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            return arr
    except Exception:
        pass

    # Try numpy.load with allow_pickle to handle complex saved objects
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            # single pickled object inside array
            if arr.shape == () and arr.dtype == object:
                obj = arr.item()
                # dict of tensors/arrays
                if isinstance(obj, dict):
                    arrays = []
                    for v in obj.values():
                        if isinstance(v, torch.Tensor):
                            arrays.append(v.detach().cpu().numpy())
                        elif isinstance(v, np.ndarray):
                            arrays.append(v)
                    if arrays:
                        return np.concatenate(arrays, axis=0)
                # nested numpy array
                elif isinstance(obj, np.ndarray):
                    return obj
            # object-dtype array with entries that may be tensors/arrays/dicts
            elif arr.dtype == object:
                inner = []
                for a in arr:
                    if isinstance(a, torch.Tensor):
                        inner.append(a.detach().cpu().numpy())
                    elif isinstance(a, np.ndarray):
                        inner.append(a)
                    elif isinstance(a, dict):
                        for v in a.values():
                            if isinstance(v, torch.Tensor):
                                inner.append(v.detach().cpu().numpy())
                            elif isinstance(v, np.ndarray):
                                inner.append(v)
                if inner:
                    return np.concatenate(inner, axis=0)
            else:
                return arr
    except Exception:
        pass

    # Fall back to torch.load for files saved with PyTorch
    try:
        obj = torch.load(path, map_location=torch.device(device), weights_only=False)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, (list, tuple)):
            return np.stack([
                o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else np.array(o)
                for o in obj
            ])
        elif isinstance(obj, dict):
            arrays = []
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    arrays.append(v.detach().cpu().numpy())
                elif isinstance(v, np.ndarray):
                    arrays.append(v)
            if arrays:
                return np.concatenate(arrays, axis=0)
    except Exception as e:
        raise RuntimeError(f"Could not load file: {path}\n{e}")

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


######## VAE ########

def save_training_curve(nll_train, nll_val, results_path):
	"""
	Plots and saves the training and validation loss curves (NLL vs epochs) in the specified results path.
	Args:
		nll_train (array-like): Training NLL values per epoch.
		nll_val (array-like): Validation NLL values per epoch.
		results_path (str): Path to the results directory.
	"""
	results_dir = os.path.join(results_path)
	os.makedirs(results_dir, exist_ok=True)
	plot_path = os.path.join(results_dir, 'training_curve.png')

	plt.figure()
	plt.plot(range(len(nll_train)), nll_train, label='Train NLL', linewidth=2)
	plt.plot(range(len(nll_val)), nll_val, label='Validation NLL', linewidth=2)
	plt.xlabel('Epochs')
	plt.ylabel('NLL')
	plt.title('Training and Validation Loss Curves')
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_path)
	plt.close()
	print(f"Training curve saved to: {plot_path}")

def plot_nll_vs_kl(nll_train, kl_train, results_path):
	"""
	Plots the negative log-likelihood (NLL) versus KL divergence during training.
	Args:
		nll_train: List or array of training NLL values per epoch.
		kl_train: List or array of training KL values per epoch.
		results_path: Directory to save the plot.
		filename: Name of the output file (default: 'nll_vs_kl.png').
	"""
	results_dir = os.path.join(results_path)
	os.makedirs(results_dir, exist_ok=True)
	plot_path = os.path.join(results_dir, 'nll_vs_kl.png')

	plt.figure()
	plt.plot(range(len(nll_train)), nll_train, label='Train NLL', linewidth=2)
	plt.plot(range(len(kl_train)), kl_train, label='Train KL', linewidth=2)
	plt.xlabel('Epochs')
	plt.ylabel('NLL')
	plt.title('NLL vs KL during Training')
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_path)
	plt.close()
	print(f"NLL vs KL plot curve saved to: {plot_path}")

def plot_latent_tsne(model, data, label, results_path, set, perplexity=30, random_state=42):
	"""
	Projects the latent space of a VAE model using t-SNE and colors by species label.

	Args:
		model: Trained VAE model.
		data: Input spectra (numpy array).
		label: Array of string/int labels.
		results_path: Path to save the plot.
		perplexity: t-SNE perplexity parameter.
		random_state: t-SNE random seed.
	"""
	model.eval()
	with torch.no_grad():
		if torch.is_tensor(data):
			X = data.detach().clone()
		else:
			X = torch.tensor(data, dtype=torch.float32)
		mu_e, log_var_e = model.encoder.forward(X)
		latent_means = mu_e.cpu().numpy()  # shape: (N_samples, latent_dim)

	# 2. Project to 2D (t-SNE for visualization)
	latent_2d = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(latent_means)

	# 3. Convert string labels to integer codes
	if torch.is_tensor(label):
		label = label.cpu().numpy()
	unique_labels, label_numeric = np.unique(label, return_inverse=True)

	# 4. Plot, colored by label
	plt.figure(figsize=(10, 6))
	scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
				c=label_numeric, edgecolor='none', alpha=0.5,
				cmap=plt.cm.Spectral)
	plt.xlabel("Latent dim 1 (t-SNE)")
	plt.ylabel("Latent dim 2 (t-SNE)")
	plt.title("VAE latent space colored by species label")
	cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label="Class")
	cbar.ax.set_yticklabels(unique_labels)
	plt.tight_layout()
	plot_dir = os.path.join(results_path, 'plots')
	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, f'latent_tsne_{set}.png')
	plt.savefig(plot_path)
	plt.close()
	print(f"Latent t-SNE plot saved to: {plot_path}")

def plot_reconstructions(model, data, n_samples, results_path, pike_fn, random_state=42):
	"""
	Plots original and reconstructed spectra side by side for n_samples.
	Left: original spectrum, Right: reconstructed spectrum. Title shows PIKE error.

	Args:
		model: Trained VAE model.
		data: Input spectra (numpy array).
		n_samples: Number of samples to plot.
		results_path: Path to save the plot.
		pike_fn: Function to compute PIKE error (signature: pike_fn(x, x_hat)).
		random_state: Random seed for reproducibility.
	"""
	labels = data.labels
	spectra = data.data
	label_convergence = getattr(data, 'label_convergence', {})
	np.random.seed(random_state)
	idxs = np.random.choice(len(spectra), n_samples, replace=False)
	model.eval()
	with torch.no_grad():
		device = next(model.parameters()).device
		X = spectra[idxs].to(device)
		mu_e, log_var_e = model.encoder.forward(X)
		z = mu_e
		x_hat = model.decoder(z).cpu().numpy()
	fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
	for i, idx in enumerate(idxs):
		orig = data[idx][0] if len(data[idx]) > 1 else data[idx]
		recon = x_hat[i]
		pike_err = pike_fn(orig, recon)
		label = labels[idx]
		color = LABEL_TO_HEX.get(label, 'blue')
		axes[i, 0].plot(orig, color=color)
		axes[i, 0].set_title(f"Original (idx={idx})\nLabel: {label}")
		axes[i, 0].set_xlabel('m/z index')
		axes[i, 0].set_ylabel('Intensity')
		axes[i, 1].plot(recon, color=color)
		axes[i, 1].set_title(f"Reconstruction\nLabel: {label}\nPIKE={pike_err:.4f}")
		axes[i, 1].set_xlabel('m/z index')
		axes[i, 1].set_ylabel('Intensity')
	plt.tight_layout()
	plot_dir = os.path.join(results_path, 'plots')
	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, f'reconstructions_{n_samples}.png')
	plt.savefig(plot_path, dpi=600)
	plt.close()
	print(f"Reconstructions plot saved to: {plot_path}")

######## CONDITIONAL ########

def plot_latent_tsne_conditional(model, data, y_species, results_path, set, y_amr=None, perplexity=30, random_state=42):
	"""
	Projects the latent space of a ConditionalVAE model using t-SNE and colors by species label.
	Args:
		model: Trained ConditionalVAE model.
		data: Input spectra (tensor or numpy array).
		y_species: Array/tensor of species labels (int).
		results_path: Path to save the plot.
		set: Name of the dataset split (e.g. 'val', 'test').
		y_amr: Optional AMR labels (tensor or None).
		perplexity: t-SNE perplexity parameter.
		random_state: t-SNE random seed.
	"""
	model.eval()
	with torch.no_grad():
		if torch.is_tensor(data):
			X = data.detach().clone()
		else:
			X = torch.tensor(data, dtype=torch.float32)
		if torch.is_tensor(y_species):
			y_species_t = y_species.detach().clone()
		else:
			y_species_t = torch.tensor(y_species, dtype=torch.long)
		if y_amr is not None:
			if torch.is_tensor(y_amr):
				y_amr_t = y_amr.detach().clone()
			else:
				y_amr_t = torch.tensor(y_amr, dtype=torch.float32)
		else:
			y_amr_t = None
		mu_e, log_var_e = model.encoder.forward(X, y_species_t, y_amr_t)
		latent_means = mu_e.cpu().numpy()
	# 2. Project to 2D (t-SNE for visualization)
	latent_2d = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(latent_means)
	# 3. Convert labels to integer codes
	if torch.is_tensor(y_species):
		y_species_np = y_species.cpu().numpy()
	else:
		y_species_np = np.array(y_species)
	unique_labels, label_numeric = np.unique(y_species_np, return_inverse=True)
	# 4. Plot, colored by label
	plt.figure(figsize=(10, 6))
	scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
				c=label_numeric, edgecolor='none', alpha=0.5,
				cmap=plt.cm.Spectral)
	plt.xlabel("Latent dim 1 (t-SNE)")
	plt.ylabel("Latent dim 2 (t-SNE)")
	plt.title("ConditionalVAE latent space colored by species label")
	cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label="Class")
	cbar.ax.set_yticklabels(unique_labels)
	plt.tight_layout()
	plot_dir = os.path.join(results_path, 'plots')
	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, f'latent_tsne_{set}_conditional.png')
	plt.savefig(plot_path)
	plt.close()
	print(f"ConditionalVAE latent t-SNE plot saved to: {plot_path}")

def plot_reconstructions_conditional(model, data, n_samples, results_path, pike_fn, y_species, y_amr=None, random_state=42):
	"""
	Plots original and reconstructed spectra side by side for n_samples for ConditionalVAE.
	Left: original spectrum, Right: reconstructed spectrum. Title shows PIKE error.
	Args:
		model: Trained ConditionalVAE model.
		data: Dataset object with .data and .labels
		n_samples: Number of samples to plot.
		results_path: Path to save the plot.
		pike_fn: Function to compute PIKE error (signature: pike_fn(x, x_hat)).
		y_species: Array/tensor of species labels (int).
		y_amr: Optional AMR labels (tensor or None).
		random_state: Random seed for reproducibility.
	"""

	labels = data.labels
	spectra = data.data
	np.random.seed(random_state)
	idxs = np.random.choice(len(spectra), n_samples, replace=False)
	sample_labels = np.array([labels[idx] for idx in idxs])
	model.eval()
	with torch.no_grad():
		device = next(model.parameters()).device
		X = spectra[idxs].to(device)
		if torch.is_tensor(y_species):
			y_species_t = y_species[idxs].detach().clone().to(device)
		else:
			y_species_t = torch.tensor(y_species[idxs], dtype=torch.long, device=device)
		if y_amr is not None:
			if torch.is_tensor(y_amr):
				y_amr_t = y_amr[idxs].detach().clone().to(device)
			else:
				y_amr_t = torch.tensor(y_amr[idxs], dtype=torch.float32, device=device)
		else:
			y_amr_t = None
		mu_e, log_var_e = model.encoder.forward(X, y_species_t, y_amr_t)
		z = mu_e
		x_hat = model.decoder.forward(z, y_species_t, y_amr_t).cpu()
	fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
	for i, idx in enumerate(idxs):
		orig, orig_label = data[idx]
		# Ensure orig and recon are tensors
		if isinstance(orig, np.ndarray):
			orig_t = torch.tensor(orig, dtype=torch.float32)
		else:
			orig_t = orig.detach().cpu() if hasattr(orig, 'detach') else torch.tensor(orig, dtype=torch.float32)
		recon_t = x_hat[i]
		if isinstance(recon_t, np.ndarray):
			recon_t = torch.tensor(recon_t, dtype=torch.float32)
		else:
			recon_t = recon_t.detach().cpu() if hasattr(recon_t, 'detach') else torch.tensor(recon_t, dtype=torch.float32)
		pike_err = pike_fn(orig_t, recon_t)
		label = sample_labels[i]
		color = LABEL_TO_HEX.get(label, 'blue')
		axes[i, 0].plot(orig_t.numpy(), color=color)
		axes[i, 0].set_title(f"Original (idx={idx})\nLabel: {label}")
		axes[i, 0].set_xlabel('m/z index')
		axes[i, 0].set_ylabel('Intensity')
		axes[i, 1].plot(recon_t.numpy(), color=color)
		axes[i, 1].set_title(f"Reconstruction\nLabel: {label}\nPIKE={pike_err:.4f}")
		axes[i, 1].set_xlabel('m/z index')
		axes[i, 1].set_ylabel('Intensity')
	plt.tight_layout()
	plot_dir = os.path.join(results_path, 'plots')
	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, f'reconstructions_{n_samples}_conditional.png')
	plt.savefig(plot_path)
	plt.close()
	print(f"ConditionalVAE reconstructions plot saved to: {plot_path}")

def plot_generated_spectra_per_label(generated_spectra, summary_mean, label_name, n_samples, results_path):
	"""
	Plot up to n_samples generated spectra for a label in a single column (each spectrum in a row), overlaying mean and std spectrum.
	Each subplot: std fill (light grey, clipped to zero), mean (thin black), generated (label color), PIKE value in title.
	Args:
		generated_spectra: np.ndarray or torch.Tensor, shape [N, ...]
		mean_std_tuple: tuple (mean, std) for the label
		label_name: str, label name
		n_samples: int, number of generated spectra to plot
		results_path: str, directory to save the plot
	"""
	n_total = generated_spectra.shape[0]
	n_plot = min(n_samples, n_total)
	idxs = np.random.choice(n_total, n_plot, replace=False)
	sampled_spectra = generated_spectra[idxs]
	mean_spec, std_spec, max_spec, min_spec = summary_mean
	from losses.PIKE_GPU import calculate_PIKE_gpu
	def to_numpy(x):
		if hasattr(x, 'detach'):
			return x.detach().cpu().numpy()
		elif hasattr(x, 'cpu'):
			return x.cpu().numpy()
		else:
			return x
	def to_tensor(x):
		if isinstance(x, np.ndarray):
			return torch.tensor(x, dtype=torch.float32)
		elif hasattr(x, 'detach'):
			return x.detach().cpu()
		elif hasattr(x, 'cpu'):
			return x.cpu()
		else:
			return torch.tensor(x, dtype=torch.float32)
	mean_spec = to_tensor(mean_spec).squeeze().float()
	std_spec = to_tensor(std_spec).squeeze().float()
	max_spec = to_tensor(max_spec).squeeze().float()
	min_spec = to_tensor(min_spec).squeeze().float()

	color = LABEL_TO_HEX.get(label_name, 'blue')

	# PLOT WITH MEAN AND STD
	fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.5 * n_plot), sharex=True)
	if n_plot == 1:
		axes = [axes]
	x_axis = np.arange(mean_spec.shape[-1])
	lower = np.maximum(to_numpy(mean_spec - std_spec), 0)
	upper = to_numpy(mean_spec + std_spec)
	for i in range(n_plot):
		ax = axes[i]
		gen_spec = sampled_spectra[i]
		gen_spec_t = to_tensor(gen_spec)
		try:
			pike_val = calculate_PIKE_gpu(mean_spec, gen_spec_t)
		except Exception:
			pike_val = float('nan')
		ax.fill_between(x_axis, lower, upper, color='lightgrey', alpha=0.5, label='Std')
		ax.plot(x_axis, to_numpy(mean_spec), color='black', linewidth=1.0, label='Mean')
		ax.plot(x_axis, to_numpy(gen_spec), color=color, linewidth=2.0, label=f'Generated #{i+1}', alpha=0.8)
		ax.set_ylabel("Intensity")
		ax.set_title(f"{label_name} - Generated #{i+1}\nPIKE={pike_val:.4f}")
		ax.legend(loc='upper right', fontsize='small')
	axes[-1].set_xlabel("m/z index")
	plt.tight_layout()
	plot_path = os.path.join(results_path, f"Samples_{label_name}_std.png")
	plt.savefig(plot_path, dpi=600)
	plt.close(fig)
	print(f"Saved combined plot: {plot_path}")

	# PLOT WITH MEAN AND MIN/MAX
	fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.5 * n_plot), sharex=True)
	if n_plot == 1:
		axes = [axes]
	x_axis = np.arange(mean_spec.shape[-1])
	for i in range(n_plot):
		ax = axes[i]
		gen_spec = sampled_spectra[i]
		gen_spec_t = to_tensor(gen_spec)
		try:
			pike_val = calculate_PIKE_gpu(mean_spec, gen_spec_t)
		except Exception:
			pike_val = float('nan')
		ax.fill_between(x_axis, min_spec, max_spec, color='lightgrey', alpha=0.5, label='Std')
		ax.plot(x_axis, to_numpy(mean_spec), color='black', linewidth=1.0, label='Mean')
		ax.plot(x_axis, to_numpy(gen_spec), color=color, linewidth=2.0, label=f'Generated #{i+1}', alpha=0.8)
		ax.set_ylabel("Intensity")
		ax.set_title(f"{label_name} - Generated #{i+1}\nPIKE={pike_val:.4f}")
		ax.legend(loc='upper right', fontsize='small')
	axes[-1].set_xlabel("m/z index")
	plt.tight_layout()
	plot_path = os.path.join(results_path, f"Samples_{label_name}_minmax.png")
	plt.savefig(plot_path, dpi=600)
	plt.close(fig)
	print(f"Saved combined plot: {plot_path}")

def plot_joint_tsne(model, sets_dict, generated_spectra, label_name_to_index, n_samples=50, results_path=None, random_state=42, logger=None):
	"""
	Compute a joint t-SNE embedding for multiple sets (train, val, test, ood, synthetic) and plot pairwise comparisons.
	Args:
		model: Trained ConditionalVAE model.
		sets_dict: dict with keys as set names and values as dataset objects (with .data and .labels attributes).
		generated_spectra: dict with label_name -> np.ndarray of generated spectra (optional, for synthetic set).
		n_samples: Number of samples to use from each set.
		results_path: Path to save the plots.
		random_state: Random seed for reproducibility.
	"""

	model.eval()
	device = next(model.parameters()).device
	all_latents = []
	all_labels = []
	all_domains = []

	# Collect latent means for each set, sampling up to n_samples per label per domain
	for domain, dataset in sets_dict.items():
		X = dataset.data.to(device)
		y = dataset.labels.to(device)
		domain_latents = []
		domain_labels = []
		domain_domains = []
		unique_labels = torch.unique(y)
		for label in unique_labels:
			label_mask = (y == label)
			label_indices = torch.where(label_mask)[0].cpu().numpy()
			n_to_sample = min(n_samples, len(label_indices))
			if n_to_sample == 0:
				continue
			np.random.seed(random_state)
			chosen = np.random.choice(label_indices, n_to_sample, replace=False)
			X_label = X[chosen]
			y_label = y[chosen]
			with torch.no_grad():
				mu_e, _ = model.encoder(X_label, y_label)
			domain_latents.append(mu_e.cpu().numpy())
			domain_labels.append(y_label.cpu().numpy())
			domain_domains.extend([domain]*n_to_sample)
		if domain_latents:
			all_latents.append(np.concatenate(domain_latents, axis=0))
			all_labels.append(np.concatenate(domain_labels, axis=0))
			all_domains.extend(domain_domains)

	# Add generated/synthetic spectra if provided
	# Accept label_name_to_index as an argument for generated spectra
	import inspect
	frame = inspect.currentframe()
	args, _, _, values = inspect.getargvalues(frame)
	label_name_to_index = values.get('label_name_to_index', None)
	if generated_spectra is not None:
		if label_name_to_index is None:
			# Try to infer from unique labels in sets_dict
			all_labels_flat = []
			for dataset in sets_dict.values():
				all_labels_flat.extend(dataset.labels.cpu().numpy().tolist())
			unique_labels = sorted(set(all_labels_flat))
			label_name_to_index = {str(l): int(l) for l in unique_labels}
		X_gen = []
		y_gen = []
		domain_gen = []
		for label, arr in generated_spectra.items():
			n_to_sample = min(n_samples, arr.shape[0])
			if n_to_sample == 0:
				continue
			np.random.seed(random_state)
			chosen = np.random.choice(arr.shape[0], n_to_sample, replace=False)
			arr_sampled = arr[chosen]
			X_gen.append(torch.tensor(arr_sampled, dtype=torch.float32))
			idx = label_name_to_index[str(label)] if str(label) in label_name_to_index else 0
			y_gen.extend([idx]*n_to_sample)
			domain_gen.extend(["synthetic"]*n_to_sample)
		if X_gen:
			X_gen = torch.cat(X_gen, dim=0)
			y_gen = np.array(y_gen)
			X_gen = X_gen.to(device)
			y_gen_tensor = torch.tensor(y_gen, dtype=torch.long, device=device)
			with torch.no_grad():
				mu_e, _ = model.encoder(X_gen, y_gen_tensor)
			all_latents.append(mu_e.cpu().numpy())
			all_labels.append(y_gen)
			all_domains.extend(domain_gen)

	# Stack all latents and labels
	latents = np.concatenate(all_latents, axis=0)
	labels = np.concatenate(all_labels, axis=0)
	domains = np.array(all_domains)

	# Compute joint t-SNE
	tsne = TSNE(n_components=2, perplexity=30, random_state=random_state)
	latents_2d = tsne.fit_transform(latents)

	# Save t-SNE coordinates for later use
	tsne_save_path = os.path.join(results_path, "joint_tsne_coords.npz") if results_path else "joint_tsne_coords.npz"
	np.savez(tsne_save_path, latents_2d=latents_2d, labels=labels, domains=domains)

	# Plot: color by label, marker by domain
	plot_dir = os.path.join(results_path, 'tsne') if results_path else 'tsne'
	os.makedirs(plot_dir, exist_ok=True)
	unique_label_names = np.unique(labels)
	unique_domains = list(dict.fromkeys(domains))  # preserve order
	domain_to_marker = {d: m for d, m in zip(unique_domains, ['o', 'x', 's', '^', 'D', '*', '+'])}

	plt.figure(figsize=(10, 8))
	for d in unique_domains:
		for lbl in unique_label_names:
			idx = (domains == d) & (labels == lbl)
			n_points = np.count_nonzero(idx)
			if n_points > 0:
				plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1],
							color=LABEL_TO_HEX.get(lbl, 'blue'),
							marker=domain_to_marker[d],
							label=f"{d} - {lbl}",
							alpha=0.7, s=40, edgecolor='k', linewidth=0.5)
				if logger is not None:
					logger.info(f"Plotted {n_points} samples for domain '{d}', label '{lbl}' in t-SNE plot.")
	# Custom legends: one for labels, one for domains
	from matplotlib.lines import Line2D
	label_legend = [Line2D([0], [0], marker='o', color='w', label=str(lbl),
						   markerfacecolor=LABEL_TO_HEX.get(lbl, 'blue'), markersize=10) for lbl in unique_label_names]
	domain_legend = [Line2D([0], [0], marker=domain_to_marker[d], color='k', label=d,
							markerfacecolor='w', markersize=10) for d in unique_domains]
	plt.xlabel("t-SNE dim 1")
	plt.ylabel("t-SNE dim 2")
	plt.title("Joint t-SNE: color=label, marker=domain")
	first_legend = plt.legend(handles=label_legend, title="Label", loc='upper right')
	plt.gca().add_artist(first_legend)
	plt.legend(handles=domain_legend, title="Domain", loc='lower right')
	plt.tight_layout()
	plot_path = os.path.join(plot_dir, f"joint_tsne_label_domain.png")
	plt.savefig(plot_path)
	plt.close()
	print(f"Saved t-SNE plot: {plot_path}")

def plot_tsne_from_saved(tsne_path, domains_to_plot=None, results_path=None, logger=None, n_samples=50):
	"""
	Plot t-SNE from saved coordinates, filtering by specified domains, and sampling up to n_samples per label per domain.
	Args:
		tsne_path: Path to the .npz file with keys 'latents_2d', 'labels', 'domains'.
		domains_to_plot: List of domain names to include (e.g., ['train', 'test', 'synthetic']). If None, plot all.
		results_path: Directory to save the plot. If None, saves in same dir as tsne_path.
		logger: Optional logger for info output.
		n_samples: Number of samples per label per domain (default 50).
	"""
	data = np.load(tsne_path, allow_pickle=True)
	latents_2d = data['latents_2d']
	labels = data['labels']
	domains = data['domains']
	# Convert bytes to str if needed
	if hasattr(domains[0], 'decode'):
		domains = np.array([d.decode() if hasattr(d, 'decode') else d for d in domains])
	if hasattr(labels[0], 'decode'):
		labels = np.array([l.decode() if hasattr(l, 'decode') else l for l in labels])
	# Filter by domains
	if domains_to_plot is not None:
		mask = np.isin(domains, domains_to_plot)
		latents_2d = latents_2d[mask]
		labels = labels[mask]
		domains = domains[mask]
	# Sample up to n_samples per label per domain
	all_latents = []
	all_labels = []
	all_domains = []
	unique_domains = list(dict.fromkeys(domains))
	unique_label_names = np.unique(labels)
	for d in unique_domains:
		for lbl in unique_label_names:
			idx = (domains == d) & (labels == lbl)
			idx_indices = np.where(idx)[0]
			n_to_sample = min(n_samples, len(idx_indices))
			if n_to_sample == 0:
				continue
			np.random.seed(42)
			chosen = np.random.choice(idx_indices, n_to_sample, replace=False)
			all_latents.append(latents_2d[chosen])
			all_labels.append([lbl]*n_to_sample)
			all_domains.extend([d]*n_to_sample)
			if logger is not None:
				logger.info(f"Plotted {n_to_sample} samples for domain '{d}', label '{lbl}' in t-SNE plot.")
	if not all_latents:
		print("No samples to plot for the selected domains and labels.")
		return
	latents_2d_plot = np.concatenate(all_latents, axis=0)
	labels_plot = np.concatenate(all_labels, axis=0)
	domains_plot = np.array(all_domains)
	unique_label_names_plot = np.unique(labels_plot)
	unique_domains_plot = list(dict.fromkeys(domains_plot))
	domain_to_marker = {d: m for d, m in zip(unique_domains_plot, ['o', 'x', 's', '^', 'D', '*', '+'])}

	plt.figure(figsize=(10, 8))
	for d in unique_domains_plot:
		for lbl in unique_label_names_plot:
			idx = (domains_plot == d) & (labels_plot == lbl)
			n_points = np.count_nonzero(idx)
			if n_points > 0:
				plt.scatter(latents_2d_plot[idx, 0], latents_2d_plot[idx, 1],
							color=LABEL_TO_HEX.get(lbl, 'blue'),
							marker=domain_to_marker[d],
							label=f"{d} - {lbl}",
							alpha=0.7, s=40, edgecolor='k', linewidth=0.5)
	# Custom legends: one for labels, one for domains
	from matplotlib.lines import Line2D
	label_legend = [Line2D([0], [0], marker='o', color='w', label=str(lbl),
						   markerfacecolor=LABEL_TO_HEX.get(lbl, 'blue'), markersize=10) for lbl in unique_label_names_plot]
	domain_legend = [Line2D([0], [0], marker=domain_to_marker[d], color='k', label=d,
							markerfacecolor='w', markersize=10) for d in unique_domains_plot]
	plt.xlabel("t-SNE dim 1")
	plt.ylabel("t-SNE dim 2")
	plt.title("t-SNE: color=label, marker=domain")
	first_legend = plt.legend(handles=label_legend, title="Label", loc='upper right')
	plt.gca().add_artist(first_legend)
	plt.legend(handles=domain_legend, title="Domain", loc='lower right')
	plt.tight_layout()
	# Save
	plot_dir = os.path.join(results_path, 'tsne') if results_path else 'tsne'
	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, f"tsne_domains_{'_'.join(domains_to_plot) if domains_to_plot else 'all'}.png")
	plt.savefig(plot_path)
	plt.close()
	print(f"Saved t-SNE plot for domains {domains_to_plot if domains_to_plot else 'all'}: {plot_path}")

def plot_tsne_real_vs_synth(model, generated_data, data, results_path, n_real_per_label=500, n_synth=2000, random_state=42):
		"""
		Visualize t-SNE of real vs synthetic spectra in VAE latent space.
		Args:
			generated_data: Synthetic spectra (numpy array or torch.Tensor)
			data: Dataset object with .data and .labels attributes (real data)
			results_path: Directory to save the plot
			n_real_per_label: Number of real spectra per label to sample
			n_synth: Number of synthetic spectra to sample
			random_state: Random seed
		"""
		device = next(model.parameters()).device

		# 1. Sample real spectra: n_real_per_label per label
		real_data = data.data.cpu().numpy() if hasattr(data.data, 'cpu') else data.data
		real_labels = data.labels.cpu().numpy() if hasattr(data.labels, 'cpu') else data.labels
		unique_labels = np.unique(real_labels)
		idxs = []
		for label in unique_labels:
			label_idxs = np.where(real_labels == label)[0]
			np.random.seed(random_state)
			chosen = np.random.choice(label_idxs, min(n_real_per_label, len(label_idxs)), replace=False)
			idxs.extend(chosen)
		real_sampled = real_data[idxs]
		real_sampled_labels = real_labels[idxs]

		# 2. Sample synthetic spectra
		synth_data = generated_data.cpu().numpy() if hasattr(generated_data, 'cpu') else generated_data
		if synth_data.shape[0] > n_synth:
			np.random.seed(random_state)
			chosen = np.random.choice(synth_data.shape[0], n_synth, replace=False)
			synth_sampled = synth_data[chosen]
		else:
			synth_sampled = synth_data

		# 3. Get latent means for both
		with torch.no_grad():
			real_tensor = torch.tensor(real_sampled, dtype=torch.float32).to(device)
			mu_real, _ = model.encoder.forward(real_tensor)
			mu_real = mu_real.cpu().numpy()
			synth_tensor = torch.tensor(synth_sampled, dtype=torch.float32).to(device)
			mu_synth, _ = model.encoder.forward(synth_tensor)
			mu_synth = mu_synth.cpu().numpy()

		# 4. Concatenate and run t-SNE
		all_latent = np.concatenate([mu_real, mu_synth], axis=0)
		tsne = TSNE(n_components=2, random_state=random_state)
		latent_2d = tsne.fit_transform(all_latent)

		# 5. Plot
		plt.figure(figsize=(10, 8))
		# Real: color by label using Spectral colormap
		# Assign integer codes to labels for colormap
		unique_labels = np.unique(real_sampled_labels)
		label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
		label_numeric = np.array([label_to_int[lbl] for lbl in real_sampled_labels])
		scatter = plt.scatter(latent_2d[:len(mu_real), 0], latent_2d[:len(mu_real), 1],
			c=label_numeric, edgecolor='none', alpha=0.5, cmap=plt.cm.Spectral, s=40)
		# Synthetic: single color (blue)
		plt.scatter(latent_2d[len(mu_real):, 0], latent_2d[len(mu_real):, 1],
			c='blue', alpha=0.8, label='Synthetic', s=40, edgecolor='none')
		plt.xlabel('t-SNE dim 1')
		plt.ylabel('t-SNE dim 2')
		plt.title('VAE latent t-SNE: Real (colored by label) vs Synthetic (blue)')
		# Custom legend for real labels (use printed_names, italic mathtext)
		from matplotlib.lines import Line2D
		# Legend: use colormap for markerfacecolor
		import matplotlib.cm as cm
		norm = plt.Normalize(vmin=0, vmax=len(unique_labels)-1)
		label_legend = [Line2D([0], [0], marker='o', color='w',
			label=f"$\\it{{{printed_names.get(str(lbl), str(lbl))}}}$",
			markerfacecolor=cm.Spectral(norm(i)), markersize=10) for i, lbl in enumerate(unique_labels)]
		plt.legend(handles=label_legend + [Line2D([0], [0], marker='o', color='w', label='Synthetic',
			markerfacecolor='blue', markersize=10)],
			title="Label", loc='upper right')
		plot_dir = os.path.join(results_path, 'plots')
		os.makedirs(plot_dir, exist_ok=True)
		plot_path = os.path.join(plot_dir, 'latent_tsne_real_vs_synth.png')
		plt.tight_layout()
		plt.savefig(plot_path, dpi=600)
		plt.close()
		print(f"Latent t-SNE (real vs synthetic) plot saved to: {plot_path}")


######## GAN ########
def training_curve_gan(history, results_path):
    """
    Plot GAN training and validation losses for both Generator and Discriminator.

    Args:
        history (dict): contains loss lists with keys:
            - "D_train": list of discriminator training losses
            - "G_train": list of generator training losses
            - "D_val": list of discriminator validation losses
            - "G_val": list of generator validation losses
        results_path (str): directory to save the plot.
    """
    os.makedirs(results_path, exist_ok=True)

    epochs_range = range(1, len(history["D_train"]) + 1)

    plt.figure(figsize=(8, 5))

    # Discriminator losses (same color, different line style)
    plt.plot(epochs_range, history["D_train"], color="tab:blue", label="Discriminator (train)")
    plt.plot(epochs_range, history["D_val"], color="tab:blue", linestyle="--", label="Discriminator (val)")

    # Generator losses (same color, different line style)
    plt.plot(epochs_range, history["G_train"], color="tab:orange", label="Generator (train)")
    plt.plot(epochs_range, history["G_val"], color="tab:orange", linestyle="--", label="Generator (val)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training and Validation Losses")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(results_path, "GAN_training_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved training curve to: {save_path}")

def plot_meanVSgenerated(generated_sample, mean_spectra, results_path):
	"""
	Plot one generated spectrum per label over its mean and std real spectrum (6 subplots total).
	Each subplot shows: std fill (light grey, clipped to zero), mean (thin black), generated (label color), PIKE value in title.
	"""
	from losses.PIKE_GPU import calculate_PIKE_gpu
	def to_numpy(x):
		if hasattr(x, 'detach'):
			return x.detach().cpu().numpy()
		elif hasattr(x, 'cpu'):
			return x.cpu().numpy()
		else:
			return x
	def to_tensor(x):
		if isinstance(x, np.ndarray):
			return torch.tensor(x, dtype=torch.float32)
		elif hasattr(x, 'detach'):
			return x.detach().cpu()
		elif hasattr(x, 'cpu'):
			return x.cpu()
		else:
			return torch.tensor(x, dtype=torch.float32)

	fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True, sharey=True)
	if not isinstance(axes, np.ndarray):
		axes = [axes]

	labels = mean_spectra.keys()
	for i, label_name in enumerate(labels):
		ax = axes[i]
		color = LABEL_TO_HEX.get(label_name, 'blue')
		# mean_spectra is now a dict: label → (mean, std)
		mean_spec, std_spec = mean_spectra[i]
		mean_spec = to_tensor(mean_spec).squeeze().float()
		std_spec = to_tensor(std_spec).squeeze().float()

		pike_val = calculate_PIKE_gpu(mean_spec, to_tensor(generated_sample))

		x_axis = np.arange(mean_spec.shape[-1])
		lower = np.maximum(to_numpy(mean_spec - std_spec), 0)
		upper = to_numpy(mean_spec + std_spec)
		ax.fill_between(x_axis, lower, upper, color='lightgrey', alpha=0.5, label='Std')
		ax.plot(x_axis, to_numpy(mean_spec), color='black', linewidth=1.0, label='Mean')
		ax.plot(x_axis, to_numpy(generated_sample), color=color, linewidth=2.0, label='Generated', alpha=0.8)

		ax.set_title(f"{label_name}\nPIKE={pike_val:.4f}")
		ax.legend(loc='upper right', fontsize='small')
		ax.set_xticks([])
		ax.set_yticks([])

	for k in range(len(labels), len(axes)):
		axes[k].axis('off')

	axes[-1].set_xlabel("m/z index")
	fig.suptitle("Generated Spectra vs Mean/Std Real Spectra (1 per label)", fontsize=16)
	plt.tight_layout()
	plt.savefig(results_path, dpi=300)
	plt.close(fig)
	print(f"Saved combined plot: {results_path}")


######## DM ########
def plot_generated_vs_all_means(generated_sample, summary_spectra, label_correspondence, plots_path, logger):
	"""
	Plot a single generated spectrum vs. the mean and std spectra of all labels (6 subplots).
	Each subplot: colored = generated, thin black = mean, light grey fill = std.
	"""
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	generated_sample = generated_sample.to(device).float()

	# Order labels by their defined sequence
	labels = summary_spectra.keys()
	ordered_items = sorted(summary_spectra.items(), key=lambda x: list(label_correspondence.keys()).index(x[0]))

	# PLOT WITH MEAN AND STD
	fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2.5 * len(labels)), sharex=True)
	for i, (label_id, (mean_spec, std_spec, _, _)) in enumerate(ordered_items):
		label_name = label_correspondence[label_id]
		color = LABEL_TO_HEX.get(label_name, 'C0')
		ax = axes[i]
		mean_spec = mean_spec.squeeze().to(device).float()
		std_spec = std_spec.squeeze().to(device).float()

		# Compute PIKE distance between mean and generated spectrum
		try:
			pike_val = calculate_PIKE_gpu(mean_spec, generated_sample)
		except Exception:
			pike_val = float('nan')

		x_axis = np.arange(mean_spec.shape[-1])
		# Clip lower bound to zero
		lower = np.maximum((mean_spec - std_spec).cpu().numpy(), 0)
		upper = (mean_spec + std_spec).cpu().numpy()
		ax.fill_between(x_axis,
				lower,
				upper,
				color='lightgrey', alpha=0.5, label='Std')
		# Plot mean (thin black)
		ax.plot(x_axis, mean_spec.cpu().numpy(), color='black', linewidth=1.0, label=f'Mean {label_name}')
		# Plot generated (label color)
		ax.plot(x_axis, generated_sample.detach().cpu().numpy(), color=color, linewidth=2.0, label='Generated', alpha=0.8)

		ax.set_ylabel("Intensity", fontsize=9)
		ax.set_title(f"{label_name}  (PIKE={pike_val:.4f})", fontsize=10)
		ax.legend(loc='upper right', fontsize='x-small')

	axes[-1].set_xlabel("m/z index")
	plt.tight_layout()
	save_path = plots_path + '_means_std.png'
	plt.savefig(save_path, dpi=600)
	plt.close(fig)
	logger.info(f"✅ Saved combined comparison plot → {save_path}")

	# PLOT WITH MEAN AND MIN/MAX
	fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2.5 * len(labels)), sharex=True)
	for i, (label_id, (mean_spec, _, max_spec, min_spec)) in enumerate(ordered_items):
		label_name = label_correspondence[label_id]
		color = LABEL_TO_HEX.get(label_name, 'C0')
		ax = axes[i]
		mean_spec = mean_spec.squeeze().to(device).float()
		std_spec = std_spec.squeeze().to(device).float()

		# Compute PIKE distance between mean and generated spectrum
		try:
			pike_val = calculate_PIKE_gpu(mean_spec, generated_sample)
		except Exception:
			pike_val = float('nan')

		x_axis = np.arange(mean_spec.shape[-1])
		ax.fill_between(x_axis,
				min_spec.cpu().numpy().squeeze(),
				max_spec.cpu().numpy().squeeze(),
				color='lightgrey', alpha=0.5, label='MinMax')
		# Plot mean (thin black)
		ax.plot(x_axis, mean_spec.cpu().numpy(), color='black', linewidth=1.0, label=f'Mean {label_name}')
		# Plot generated (label color)
		ax.plot(x_axis, generated_sample.detach().cpu().numpy(), color=color, linewidth=2.0, label='Generated', alpha=0.8)

		ax.set_ylabel("Intensity", fontsize=9)
		ax.set_title(f"{label_name}  (PIKE={pike_val:.4f})", fontsize=10)
		ax.legend(loc='upper right', fontsize='x-small')

	axes[-1].set_xlabel("m/z index")
	plt.tight_layout()
	save_path = plots_path + '_means_minmax.png'
	plt.savefig(save_path, dpi=600)
	plt.close(fig)
	logger.info(f"✅ Saved combined comparison plot → {save_path}")
