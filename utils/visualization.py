import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_latent_tsne(model, data, label, results_path, perplexity=30, random_state=42):
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
		X = torch.tensor(data, dtype=torch.float32)
		mu_e, log_var_e = model.encoder.encode(X)
		latent_means = mu_e.cpu().numpy()  # shape: (N_samples, latent_dim)

	# 2. Project to 2D (t-SNE for visualization)
	latent_2d = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(latent_means)

	# 3. Convert string labels to integer codes
	unique_labels, label_numeric = np.unique(label, return_inverse=True)

	# Print equivalences
	for idx, label in enumerate(unique_labels):
		print(f"{idx}: {label}")

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
	plot_path = os.path.join(plot_dir, 'latent_tsne.png')
	plt.savefig(plot_path)
	plt.close()
	print(f"Latent t-SNE plot saved to: {plot_path}")


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
	data = data.data
	np.random.seed(random_state)
	idxs = np.random.choice(len(data), n_samples, replace=False)
	sample_labels = np.array([labels[idx] for idx in idxs])
	unique_labels = np.unique(labels)
	label_to_color = {lbl: plt.cm.Spectral(i / max(len(unique_labels)-1,1)) for i, lbl in enumerate(unique_labels)}
	model.eval()
	with torch.no_grad():
		X = torch.tensor(data[idxs], dtype=torch.float32)
		mu_e, log_var_e = model.encoder.encode(X)
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
		axes[i, 1].plot(recon, color=color)
		axes[i, 1].set_title(f"Reconstruction\nLabel: {label}\nPIKE={pike_err:.4f}")
		for ax in axes[i]:
			ax.set_xticks([])
			ax.set_yticks([])
	plt.tight_layout()
	plot_dir = os.path.join(results_path, 'plots')
	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, f'reconstructions_{n_samples}.png')
	plt.savefig(plot_path)
	plt.close()
	print(f"Reconstructions plot saved to: {plot_path}")