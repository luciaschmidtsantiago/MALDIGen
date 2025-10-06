import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


######## VAE ########

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
	unique_labels = np.unique(labels)
	label_to_color = {lbl: plt.cm.Spectral(i / max(len(unique_labels)-1,1)) for i, lbl in enumerate(unique_labels)}
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
		label = sample_labels[i]
		color = label_to_color[label] if label_to_color is not None else 'blue'
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
	unique_labels = np.unique(labels)
	label_to_color = {lbl: plt.cm.Spectral(i / max(len(unique_labels)-1,1)) for i, lbl in enumerate(unique_labels)}
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
		label = sample_labels[i]
		color = label_to_color[label] if label_to_color is not None else 'blue'
		axes[i, 0].plot(orig, color=color)
		axes[i, 0].set_title(f"Original (idx={idx})\nLabel: {orig_label}")
		axes[i, 0].set_xlabel('m/z index')
		axes[i, 0].set_ylabel('Intensity')
		axes[i, 1].plot(recon, color=color)
		axes[i, 1].set_title(f"Reconstruction\nLabel: {y_species[idx]}\nPIKE={pike_err:.4f}")
		axes[i, 1].set_xlabel('m/z index')
		axes[i, 1].set_ylabel('Intensity')
	plt.tight_layout()
	plot_dir = os.path.join(results_path, 'plots')
	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, f'reconstructions_{n_samples}_conditional.png')
	plt.savefig(plot_path)
	plt.close()
	print(f"ConditionalVAE reconstructions plot saved to: {plot_path}")