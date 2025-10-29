import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


########## FIXED COLOR MAP FOR 6 LABELS ##########
# Replace with your actual label names in the correct order:
LABEL_NAMES = [
	'Enterobacter_cloacae_complex',
	'Enterococcus_Faecium',
	'Escherichia_Coli',
	'Klebsiella_Pneumoniae',
	'Pseudomonas_Aeruginosa',
	'Staphylococcus_Aureus',
]
LABEL_TO_COLOR = {lbl: plt.cm.Spectral(i / 5) for i, lbl in enumerate(LABEL_NAMES)}

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
	np.random.seed(random_state)
	idxs = np.random.choice(len(spectra), n_samples, replace=False)
	sample_labels = np.array([labels[idx] for idx in idxs])
	# Map numeric labels to names if needed
	if np.issubdtype(sample_labels.dtype, np.integer):
		# Assume label order matches LABEL_NAMES
		label_names = [LABEL_NAMES[int(l)] if int(l) < len(LABEL_NAMES) else str(l) for l in sample_labels]
	else:
		label_names = sample_labels
	label_to_color = LABEL_TO_COLOR
	model.eval()
	with torch.no_grad():
		device = next(model.parameters()).device
		X = spectra[idxs].to(device)
		mu_e, log_var_e = model.encoder.forward(X)
		z = mu_e
		x_hat = model.decoder(z).cpu().numpy()
	fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
	for i, idx in enumerate(idxs):
		orig = data[idx]
		recon = x_hat[i]
		pike_err = pike_fn(orig, recon)
		label = label_names[i]
		color = label_to_color[label] if label in label_to_color else 'blue'
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
	plt.savefig(plot_path)
	plt.close()
	print(f"Reconstructions plot saved to: {plot_path}")

## CONDITIONAL ##

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
	# Map numeric labels to names if needed
	if np.issubdtype(sample_labels.dtype, np.integer):
		label_names = [LABEL_NAMES[int(l)] if int(l) < len(LABEL_NAMES) else str(l) for l in sample_labels]
	else:
		label_names = sample_labels
	label_to_color = LABEL_TO_COLOR
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
		x_hat = model.decoder.forward(z, y_species_t, y_amr_t).cpu().numpy()
	fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
	for i, idx in enumerate(idxs):
		orig, orig_label = data[idx]
		recon = x_hat[i]
		pike_err = pike_fn(orig, recon)
		label = label_names[i]
		color = label_to_color[label] if label in label_to_color else 'blue'
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
	plot_path = os.path.join(plot_dir, f'reconstructions_{n_samples}_conditional.png')
	plt.savefig(plot_path)
	plt.close()
	print(f"ConditionalVAE reconstructions plot saved to: {plot_path}")

def plot_generated_spectra_per_label(generated_spectra, mean_spectrum, label_name, n_samples, results_path):
	"""
	Plot up to n_samples generated spectra for a label in a single column (each spectrum in a row), overlaying the mean spectrum.
	"""
	n_total = generated_spectra.shape[0]
	n_plot = min(n_samples, n_total)
	# Randomly sample n_plot indices
	idxs = np.random.choice(n_total, n_plot, replace=False)
	sampled_spectra = generated_spectra[idxs]
	fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.5 * n_plot), sharex=True)
	if n_plot == 1:
		axes = [axes]
	from losses.PIKE_GPU import calculate_PIKE_gpu
	# Use fixed color map for label
	label_to_color = LABEL_TO_COLOR
	for i in range(n_plot):
		ax = axes[i]
		# Ensure data is on CPU and numpy for matplotlib, but keep torch for PIKE
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
		gen_spec = sampled_spectra[i]
		mean_spec = mean_spectrum
		# Calculate PIKE value
		gen_spec_t = to_tensor(gen_spec)
		mean_spec_t = to_tensor(mean_spec)
		try:
			pike_val = calculate_PIKE_gpu(gen_spec_t, mean_spec_t)
		except Exception as e:
			pike_val = float('nan')
		# Color for this label
		color = label_to_color.get(label_name, 'blue')
		# Plot
		ax.plot(to_numpy(gen_spec), label=f'Generated #{i+1}', color=color)
		ax.plot(to_numpy(mean_spec), label='Mean spectrum', color='gray', linestyle='--', alpha=0.4, linewidth=2)
		ax.set_ylabel("Intensity")
		ax.set_title(f"{label_name} - Generated #{i+1}\nPIKE={pike_val:.4f}")
		ax.legend(loc='upper right', fontsize='small')
	axes[-1].set_xlabel("m/z index")
	plt.tight_layout()
	plot_path = os.path.join(results_path, f"Samples_{label_name}.png")
	plt.savefig(plot_path)
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

	# Map numeric labels to names if needed
	if np.issubdtype(labels.dtype, np.integer):
		label_names = np.array([LABEL_NAMES[int(l)] if int(l) < len(LABEL_NAMES) else str(l) for l in labels])
	else:
		label_names = labels

	# Compute joint t-SNE
	tsne = TSNE(n_components=2, perplexity=30, random_state=random_state)
	latents_2d = tsne.fit_transform(latents)

	# Save t-SNE coordinates for later use
	tsne_save_path = os.path.join(results_path, "joint_tsne_coords.npz") if results_path else "joint_tsne_coords.npz"
	np.savez(tsne_save_path, latents_2d=latents_2d, labels=label_names, domains=domains)

	# Plot: color by label, marker by domain
	plot_dir = os.path.join(results_path, 'tsne') if results_path else 'tsne'
	os.makedirs(plot_dir, exist_ok=True)
	unique_label_names = np.unique(label_names)
	unique_domains = list(dict.fromkeys(domains))  # preserve order
	label_to_color = LABEL_TO_COLOR
	domain_to_marker = {d: m for d, m in zip(unique_domains, ['o', 'x', 's', '^', 'D', '*', '+'])}

	plt.figure(figsize=(10, 8))
	for d in unique_domains:
		for lbl in unique_label_names:
			idx = (domains == d) & (label_names == lbl)
			n_points = np.count_nonzero(idx)
			if n_points > 0:
				plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1],
							color=label_to_color.get(lbl, 'blue'),
							marker=domain_to_marker[d],
							label=f"{d} - {lbl}",
							alpha=0.7, s=40, edgecolor='k', linewidth=0.5)
				if logger is not None:
					logger.info(f"Plotted {n_points} samples for domain '{d}', label '{lbl}' in t-SNE plot.")
	# Custom legends: one for labels, one for domains
	from matplotlib.lines import Line2D
	label_legend = [Line2D([0], [0], marker='o', color='w', label=str(lbl),
						   markerfacecolor=label_to_color.get(lbl, 'blue'), markersize=10) for lbl in unique_label_names]
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

# Generic function to plot t-SNE from saved coordinates for selected domains

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
	label_to_color = LABEL_TO_COLOR
	domain_to_marker = {d: m for d, m in zip(unique_domains_plot, ['o', 'x', 's', '^', 'D', '*', '+'])}

	plt.figure(figsize=(10, 8))
	for d in unique_domains_plot:
		for lbl in unique_label_names_plot:
			idx = (domains_plot == d) & (labels_plot == lbl)
			n_points = np.count_nonzero(idx)
			if n_points > 0:
				plt.scatter(latents_2d_plot[idx, 0], latents_2d_plot[idx, 1],
							color=label_to_color.get(lbl, 'blue'),
							marker=domain_to_marker[d],
							label=f"{d} - {lbl}",
							alpha=0.7, s=40, edgecolor='k', linewidth=0.5)
	# Custom legends: one for labels, one for domains
	from matplotlib.lines import Line2D
	label_legend = [Line2D([0], [0], marker='o', color='w', label=str(lbl),
						   markerfacecolor=label_to_color.get(lbl, 'blue'), markersize=10) for lbl in unique_label_names_plot]
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
	Plot one generated spectrum per label over its mean real spectrum (6 subplots total).
	Each subplot shows the mean spectrum (color) and generated spectrum (light gray),
	and displays the PIKE value in the title.
	"""
	from losses.PIKE_GPU import calculate_PIKE_gpu
	# Local helper functions (same as your version)
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


	# === Plot setup: 6 rows, 1 column ===
	fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True, sharey=True)
	if not isinstance(axes, np.ndarray):
		axes = [axes]

	for i, label_name in enumerate(LABEL_NAMES):
		ax = axes[i]
		color = LABEL_TO_COLOR.get(label_name, 'blue')

		mean_spec = mean_spectra[i]

		pike_val = calculate_PIKE_gpu(mean_spec, generated_sample)

		# Plot generated spectrum (light gray)
		ax.plot(to_numpy(generated_sample), color='lightgray', linewidth=1.5, label='Generated')

		# Plot mean spectrum (label color)
		ax.plot(to_numpy(mean_spec), color=color, linewidth=2.5, label='Mean spectrum')

		ax.set_title(f"{label_name}\nPIKE={pike_val:.4f}")
		ax.legend(loc='upper right', fontsize='small')
		ax.set_xticks([])
		ax.set_yticks([])

	# Turn off any unused axes (if <6)
	for k in range(len(LABEL_NAMES), len(axes)):
		axes[k].axis('off')

	axes[-1].set_xlabel("m/z index")
	fig.suptitle("Generated Spectra vs Mean Real Spectra (1 per label)", fontsize=16)
	plt.tight_layout()
	plt.savefig(results_path, dpi=300)
	plt.close(fig)
	print(f"Saved combined plot: {results_path}")



######## DM ########
from losses.PIKE_GPU import calculate_PIKE_gpu
def plot_generated_vs_all_means(generated_sample, mean_spectra_dict, label_correspondence, save_path, logger):
    """
    Plot a single generated spectrum vs. the mean spectra of all labels (6 subplots).
    Each subplot: black dashed = generated, colored line = class mean.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ensure shapes
    if generated_sample.ndim == 3:
        generated_sample = generated_sample.squeeze(0).squeeze(0)
    elif generated_sample.ndim == 2:
        generated_sample = generated_sample.squeeze(0)
    generated_sample = generated_sample.to(device).float()

    # Order labels by their defined sequence (LABEL_NAMES)
    ordered_items = sorted(mean_spectra_dict.items(), key=lambda kv: LABEL_NAMES.index(label_correspondence[kv[0]]))

    fig, axes = plt.subplots(len(LABEL_NAMES), 1, figsize=(10, 2.5 * len(LABEL_NAMES)), sharex=True)

    if len(LABEL_NAMES) == 1:
        axes = [axes]

    for i, (label_id, mean_spec) in enumerate(ordered_items):
        label_name = label_correspondence[label_id]
        color = LABEL_TO_COLOR.get(label_name, 'C0')
        ax = axes[i]

        mean_spec = mean_spec.squeeze().to(device).float()

        # Compute PIKE distance between mean and generated spectrum
        try:
            pike_val = calculate_PIKE_gpu(mean_spec, generated_sample)
        except Exception:
            pike_val = float('nan')

        # Plot mean (colored) and generated (gray dashed)
        ax.plot(generated_sample.cpu().numpy(), color='lightgray', linestyle='--', linewidth=1.2, label='Generated')
        ax.plot(mean_spec.cpu().numpy(), color=color, linewidth=2.0, label=f'Mean {label_name}', alpha=0.7)

        ax.set_ylabel("Intensity", fontsize=9)
        ax.set_title(f"{label_name}  (PIKE={pike_val:.4f})", fontsize=10)
        ax.legend(loc='upper right', fontsize='x-small')

    axes[-1].set_xlabel("m/z index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"✅ Saved combined comparison plot → {save_path}")
