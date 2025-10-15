import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

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

########## FIXED MARKER MAP FOR DOMAINS ##########
DOMAIN_TO_MARKER = {
    'DRIAMS': 'o',      # circle
    'MARISMa': 'x',     # cross
    'RKI': 's',         # empty square
}

class MALDI(Dataset):
    """
    PyTorch Dataset for MALDI-TOF spectra.
    Each sample is one spectrum, optionally normalized to [0,1].
    """
    def __init__(self, data, labels, normalization=True, get_labels=False):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.normalization = normalization
        self.get_labels = get_labels

        if isinstance(labels[0], str):
            label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
            labels = np.vectorize(label_map.get)(labels)
            self.label_convergence = {v: str(k) for k, v in label_map.items()}  # int → string
        else:
            self.label_convergence = None

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrum = self.data[idx]
        if self.normalization:
            min_val, max_val = spectrum.min(), spectrum.max()
            spectrum = (spectrum - min_val) / (max_val - min_val + 1e-8)
        if self.get_labels:
            label = self.labels[idx]
            return spectrum, label
        else:
            return spectrum

def plot_tsne_flexible(embeddings, meta, color_by='domain',
                       include=None, exclude=None, title=None, save_name=None):
    """
    Flexible t-SNE plotting function.
    Args:
        embeddings: np.ndarray, shape (N, 2)
        meta: list or np.ndarray of dicts, length N
        color_by: str, metadata key to color by (e.g. 'domain', 'year', 'species', 'hospital', etc.)
        include: dict, e.g. {'domain': ['MARISMa', 'DRIAMS'], 'year': ['2018']} (only include these values)
        exclude: dict, e.g. {'domain': ['RKI']} (exclude these values)
        title: str, plot title
        save_name: str, if given, save the plot to this path
    """
    meta_arr = np.array(meta, dtype=object) if not isinstance(meta, np.ndarray) else meta
    mask = np.ones(len(meta_arr), dtype=bool)
    # Apply include filter
    if include:
        for k, v in include.items():
            mask &= np.isin([m.get(k, None) for m in meta_arr], v)
    # Apply exclude filter
    if exclude:
        for k, v in exclude.items():
            mask &= ~np.isin([m.get(k, None) for m in meta_arr], v)
    filtered_embeddings = embeddings[mask]
    filtered_meta = meta_arr[mask]
    color_values = np.array([m.get(color_by, 'Unknown') for m in filtered_meta])
    unique_values = np.unique(color_values)
    # Use fixed color map for label/species
    use_fixed = color_by.lower() in ['label', 'species']
    plt.figure(figsize=(10, 8))
    # Use domain for marker if available
    meta_domains = np.array([m.get('domain', None) for m in filtered_meta])
    for i, val in enumerate(unique_values):
        idxs = color_values == val
        # Determine marker by domain if possible
        if 'domain' in filtered_meta[0]:
            # If plotting by label/species, use domain for marker
            marker = DOMAIN_TO_MARKER.get(meta_domains[idxs][0], 'o') if np.any(idxs) else 'o'
        else:
            marker = 'o'
        if use_fixed and val in LABEL_TO_COLOR:
            color = LABEL_TO_COLOR[val]
        else:
            if len(unique_values) <= 10:
                color = cm.tab10(i / max(1, len(unique_values)-1))
            elif len(unique_values) <= 20:
                color = cm.tab20(i / max(1, len(unique_values)-1))
            else:
                color = cm.nipy_spectral(i / max(1, len(unique_values)-1))
        plt.scatter(filtered_embeddings[idxs, 0], filtered_embeddings[idxs, 1],
                    label=str(val), color=color, marker=marker, alpha=0.6, s=10)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if title:
        plt.title(title)
    else:
        plt.title(f"t-SNE colored by {color_by}")
    plt.tight_layout()
    save_name = save_name if save_name else f"tsne_colored_by_{color_by}.png"
    save_path = os.path.join(os.path.dirname(__file__), 'tsne', save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_tsne_flexible_color_shape(embeddings, meta, color_by='year', shape_by='label', include=None, exclude=None, title=None, save_name=None):
    """
    Flexible t-SNE plotting function with color and shape encoding.
    Args:
        embeddings: np.ndarray, shape (N, 2)
        meta: list or np.ndarray of dicts, length N
        color_by: str, metadata key to color by (e.g. 'year')
        shape_by: str, metadata key to shape by (e.g. 'label')
        include: dict, filter to include only certain values
        exclude: dict, filter to exclude certain values
        title: str, plot title
        save_name: str, if given, save the plot to this path
    """
    import itertools
    meta_arr = np.array(meta, dtype=object) if not isinstance(meta, np.ndarray) else meta
    mask = np.ones(len(meta_arr), dtype=bool)
    if include:
        for k, v in include.items():
            mask &= np.isin([m.get(k, None) for m in meta_arr], v)
    if exclude:
        for k, v in exclude.items():
            mask &= ~np.isin([m.get(k, None) for m in meta_arr], v)
    filtered_embeddings = embeddings[mask]
    filtered_meta = meta_arr[mask]
    color_values = np.array([m.get(color_by, 'Unknown') for m in filtered_meta])
    shape_values = np.array([m.get(shape_by, 'Unknown') for m in filtered_meta])
    unique_colors = np.unique(color_values)
    unique_shapes = np.unique(shape_values)
    # Use fixed marker map for domain if shape_by is domain
    if shape_by.lower() == 'domain':
        shape_map = {domain: DOMAIN_TO_MARKER.get(domain, 'o') for domain in unique_shapes}
    else:
        # Marker styles for up to 10 labels
        markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>']
        marker_cycle = itertools.cycle(markers)
        shape_map = {shape: next(marker_cycle) for shape in unique_shapes}
    # Use fixed color map for label/species
    use_fixed = color_by.lower() in ['label', 'species']
    color_map = {}
    for i, color in enumerate(unique_colors):
        if use_fixed and color in LABEL_TO_COLOR:
            color_map[color] = LABEL_TO_COLOR[color]
        else:
            if len(unique_colors) <= 10:
                color_map[color] = cm.tab10(i / max(1, len(unique_colors)-1))
            elif len(unique_colors) <= 20:
                color_map[color] = cm.tab20(i / max(1, len(unique_colors)-1))
            else:
                color_map[color] = cm.nipy_spectral(i / max(1, len(unique_colors)-1))
    plt.figure(figsize=(11, 9))
    for shape in unique_shapes:
        for color in unique_colors:
            idxs = (shape_values == shape) & (color_values == color)
            if np.any(idxs):
                plt.scatter(filtered_embeddings[idxs, 0], filtered_embeddings[idxs, 1],
                            label=f"{shape} | {color}",
                            color=color_map[color],
                            marker=shape_map[shape],
                            alpha=0.7, s=18)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if title:
        plt.title(title)
    else:
        plt.title(f"t-SNE colored by {color_by}, shaped by {shape_by}")
    plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    save_name = save_name if save_name else f"tsne_colored_by_{color_by}_shaped_by_{shape_by}.png"
    save_path = os.path.join(os.path.dirname(__file__), 'tsne', save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

def get_tsne(n_components=2):

    pickle_driams = "pickles/DRIAMS_study.pkl"
    pickle_marisma = "pickles/MARISMa_study.pkl"
    pickle_rki = "pickles/RKI_study.pkl"


    # --- MARISMa ---
    with open(pickle_marisma, "rb") as f:
        data_marisma = pickle.load(f)
    spectra_m, labels_m, metas_m = data_marisma['data'], data_marisma['label'], data_marisma['meta']
    years = np.array([int(m['year']) for m in metas_m])

    year_indices = {}
    for year in np.unique(years):
        year_indices[year] = np.where(years == year)[0]
    print(f"MARISMa years found: {list(year_indices.keys())}")

    # --- DRIAMS ---
    with open(pickle_driams, "rb") as f:
        data_driams = pickle.load(f)
    spectra_d, labels_d, metas_d = data_driams['data'], data_driams['label'], data_driams['meta']
    hospitals = np.array([m['hospital'] for m in metas_d])

    hospital_indices = {}
    for hospital in np.unique(hospitals):
        hospital_indices[hospital] = np.where(hospitals == hospital)[0]
    print(f"DRIAMS hospitals found: {list(hospital_indices.keys())}")

    # --- RKI ---
    with open(pickle_rki, "rb") as f:
        data_rki = pickle.load(f)
    spectra_r, labels_r, metas_r = data_rki['data'], data_rki['label'], data_rki['meta']


    # Gather all spectra and metadata for flexible t-SNE coloring
    all_spectra = []
    all_meta = []

    # MARISMa
    print("Processing MARISMa spectra...")
    for i, meta in enumerate(metas_m):
        all_spectra.append(spectra_m[i])
        all_meta.append({
            'domain': 'MARISMa',
            'year': meta['year'],
            'genus': meta['genus'],
            'species': meta['species'],
            'label': labels_m[i],
        })

    # DRIAMS
    print("Processing DRIAMS spectra...")
    for i, meta in enumerate(metas_d):
        all_spectra.append(spectra_d[i])
        all_meta.append({
            'domain': 'DRIAMS',
            'hospital': meta['hospital'],
            'genus': meta['genus'],
            'species': meta['species'],
            'label': labels_d[i],
        })

    # RKI
    print("Processing RKI spectra...")
    for i, meta in enumerate(metas_r):
        all_spectra.append(spectra_r[i])
        all_meta.append({
            'domain': 'RKI',
            'genus': meta['genus'],
            'species': meta['species'],
            'label': labels_r[i],
        })

    all_spectra = np.stack(all_spectra)



    # Use MALDI dataset class for combined data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use dummy labels for initialization (not used for coloring)
    all_dataset = MALDI(all_spectra, np.arange(len(all_spectra)), normalization=True, get_labels=True)
    data = all_dataset.data

    # Path to save/load t-SNE embeddings and metadata
    tsne_save_path = os.path.join(os.path.dirname(__file__), 'tsne', f"tsne_all_spectra_{n_components}d.npz")
    os.makedirs(os.path.dirname(tsne_save_path), exist_ok=True)

    if os.path.exists(tsne_save_path):
        print(f"Loading precomputed t-SNE from {tsne_save_path}...")
        npz = np.load(tsne_save_path, allow_pickle=True)
        embeddings = npz['embeddings']
        all_meta = npz['meta']
    else:
        print(f"Running t-SNE (n_components={n_components}) on all spectra (CPU)...")
        data_np = data.numpy()  # Always on CPU for sklearn
        tsne = TSNE(n_components=n_components, random_state=42)
        embeddings = tsne.fit_transform(data_np)
        # Save embeddings and meta
        np.savez(tsne_save_path, embeddings=embeddings, meta=np.array(all_meta, dtype=object))
        print(f"Saved t-SNE embeddings to {tsne_save_path}")

    return embeddings, all_meta

def plot_tsne_custom_groups(embeddings, meta, save_prefix="groups"):
    """
    Plot t-SNE with 4 color groups: train (blue), OOD (red), val (green), test (yellow).
    - Train: DRIAMS_A, DRIAMS_B, MARISMa (year <= 2022)
    - OOD: DRIAMS_C, DRIAMS_D
    - Val: MARISMa 2023
    - Test: MARISMa 2024
    Two subplots: right (train on top), left (OOD on top). Repeat with val/test added.
    Markers from DOMAIN_TO_MARKER.
    """
    meta_arr = np.array(meta, dtype=object) if not isinstance(meta, np.ndarray) else meta
    # Assign group
    group = []
    for m in meta_arr:
        d = m.get('domain', '')
        y = int(m.get('year', 0)) if 'year' in m else None
        if d.startswith('DRIAMS_A') or d.startswith('DRIAMS_B') or (d == 'MARISMa' and y is not None and y <= 2022):
            group.append('train')
        elif d.startswith('DRIAMS_C') or d.startswith('DRIAMS_D'):
            group.append('ood')
        elif d == 'MARISMa' and y == 2023:
            group.append('val')
        elif d == 'MARISMa' and y == 2024:
            group.append('test')
        else:
            group.append('other')
    group = np.array(group)
    color_map = {'train': 'blue', 'ood': 'red', 'val': 'green', 'test': 'yellow'}
    # Get domain for marker
    domains = np.array([m.get('domain', None) for m in meta_arr])
    years = np.array([int(m.get('year', 0)) if 'year' in m else 0 for m in meta_arr])
    # Helper to plot a group list in order
    def plot_groups(ax, group_order, alpha=0.7):
        for g in group_order:
            idxs = group == g
            for dom in np.unique(domains[idxs]):
                dom_idxs = idxs & (domains == dom)
                marker = DOMAIN_TO_MARKER.get(dom, 'o')
                ax.scatter(embeddings[dom_idxs, 0], embeddings[dom_idxs, 1],
                           c=color_map[g], marker=marker, label=f"{g} | {dom}", alpha=alpha, s=10)
    # 1. Only train and OOD
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # Right: train on top
    plot_groups(axs[1], ['ood', 'train'], alpha=0.4)  # OOD first (bottom), then train (top)
    plot_groups(axs[1], ['train'], alpha=0.8)
    axs[1].set_title("Right: Train (top), OOD (bottom)")
    # Left: OOD on top
    plot_groups(axs[0], ['train', 'ood'], alpha=0.4)
    plot_groups(axs[0], ['ood'], alpha=0.8)
    axs[0].set_title("Left: OOD (top), Train (bottom)")
    for ax in axs:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_train_ood.png", dpi=300)
    plt.close()
    # 2. Add val and test
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # Right: train, val, test, ood (train on top)
    plot_groups(axs[1], ['ood', 'val', 'test', 'train'], alpha=0.3)
    plot_groups(axs[1], ['train'], alpha=0.8)
    axs[1].set_title("Right: Train (top), OOD/Val/Test (bottom)")
    # Left: ood, val, test, train (ood on top)
    plot_groups(axs[0], ['train', 'val', 'test', 'ood'], alpha=0.3)
    plot_groups(axs[0], ['ood'], alpha=0.8)
    axs[0].set_title("Left: OOD (top), Train/Val/Test (bottom)")
    for ax in axs:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_train_ood_val_test.png", dpi=300)
    plt.close()

def main():

    tsne = True

    presaved_2D = os.path.join(os.path.dirname(__file__), 'tsne', "tsne_all_spectra_2d.npz")

    if tsne:
        print("Computing 2D t-SNE...")
        embeddings2d, all_meta2d = get_tsne(n_components=2)
    else:
        print("Loading existing 2D t-SNE...")
        #load existing 2D t-SNE
        npz2d = np.load(presaved_2D, allow_pickle=True)
        embeddings2d = npz2d['embeddings']
        all_meta2d = npz2d['meta']


    # Example: plot by domain
    plot_tsne_flexible(embeddings2d, all_meta2d, color_by='domain',
                      title="t-SNE colored by domain",
                      save_name="tsne_all_spectra_by_domain.png")

    # Example: plot by year (MARISMa only)
    include = {'domain': ['MARISMa']}
    plot_tsne_flexible(embeddings2d, all_meta2d, color_by='year',
                      include=include,
                      title="t-SNE of all spectra (MARISMa only) colored by year",
                      save_name="tsne_all_spectra_MARISMa_by_year.png")

    # Example: plot by hospital (DRIAMS only)
    include = {'domain': ['DRIAMS']}
    plot_tsne_flexible(embeddings2d, all_meta2d, color_by='hospital',
                      include=include,
                      title="t-SNE of all spectra (DRIAMS only) colored by hospital",
                      save_name="tsne_all_spectra_DRIAMS_by_hospital.png")

    # Example: plot by genus (all domains)
    plot_tsne_flexible(embeddings2d, all_meta2d, color_by='genus',
                      title="t-SNE of all spectra colored by genus",
                      save_name="tsne_all_spectra_by_genus.png")

    # Example: plot by species (all domains)
    plot_tsne_flexible(embeddings2d, all_meta2d, color_by='species',
                      title="t-SNE of all spectra colored by species",
                      save_name="tsne_all_spectra_by_species.png")

    # Example: plot by year (color) and label (shape) for MARISMa only
    include = {'domain': ['MARISMa']}
    plot_tsne_flexible_color_shape(
        embeddings2d, all_meta2d,
        color_by='year', shape_by='label',
        include=include,
        title="t-SNE of MARISMa: color=year, shape=label",
        save_name="tsne_MARISMa_by_year_and_label.png"
    )

    # Example: plot by hospital (color) and label (shape) for DRIAMS only
    include = {'domain': ['DRIAMS']}
    plot_tsne_flexible_color_shape(
        embeddings2d, all_meta2d,
        color_by='hospital', shape_by='label',
        include=include,
        title="t-SNE of DRIAMS: color=hospital, shape=label",
        save_name="tsne_DRIAMS_by_hospital_and_label.png"
    )

    # Example: plot by label (color) and domain (shape) for all domains
    plot_tsne_flexible_color_shape(
        embeddings2d, all_meta2d,
        color_by='label', shape_by='domain',
        title="t-SNE of all domains: color=label, shape=domain",
        save_name="tsne_all_domains_by_label_and_domain.png"
    )

    # Per-label t-SNE plots using plot_tsne_flexible with fixed axis limits
    labels = np.array([m.get('label', 'Unknown') for m in all_meta2d])
    unique_labels = np.unique(labels)
    # Get axis limits from the full plot
    for label in unique_labels:
        mask = labels == label
        emb = embeddings2d[mask]
        meta_label = np.array(all_meta2d, dtype=object)[mask]
        # Compose legend: domain, year, hospital if available
        legend_keys = []
        if 'domain' in meta_label[0]:
            legend_keys.append('domain')
        if 'year' in meta_label[0]:
            legend_keys.append('year')
        if 'hospital' in meta_label[0]:
            legend_keys.append('hospital')
        # Build a color_by string for legend
        if legend_keys:
            color_by = legend_keys[0]
        else:
            color_by = 'label'
        # Save name for each label
        save_name = f"tsne_per_label_{label}.png"
        title = f"t-SNE for label: {label}"
        plot_tsne_flexible(emb, meta_label, color_by=color_by, title=title, save_name=save_name)

    # Custom t-SNE plots for train/OOD/val/test groups
    plot_tsne_custom_groups(embeddings2d, all_meta2d, save_prefix=os.path.join(os.path.dirname(__file__), 'tsne', 'tsne_custom_groups'))


if __name__ == "__main__":
    main()