
# --- Fully redone script for t-SNE of all real and synthetic spectra ---
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.data import load_data

def safe_load_array(path, device="cpu"):
    try:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            return arr
    except Exception:
        pass
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            if arr.shape == () and arr.dtype == object:
                obj = arr.item()
                if isinstance(obj, dict):
                    arrays = []
                    for v in obj.values():
                        if isinstance(v, torch.Tensor):
                            arrays.append(v.detach().cpu().numpy())
                        elif isinstance(v, np.ndarray):
                            arrays.append(v)
                    if arrays:
                        return np.concatenate(arrays, axis=0)
                elif isinstance(obj, np.ndarray):
                    return obj
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

def compute_tsne(tsne_dir, dirs, model_names):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pickle_marisma = "pickles/MARISMa_study.pkl"
    pickle_driams = "pickles/DRIAMS_study.pkl"
    generated_root = "results/generated_spectra"

    # Load real data
    print("Loading real datasets...")
    train, val, test, ood = load_data(pickle_marisma, pickle_driams, logger=None, get_labels=True)
    label_convergence = train.label_convergence
    splits = {
        "train": (train.data, train.labels),
        "val": (val.data, val.labels),
        "test": (test.data, test.labels),
        "ood": (ood.data, ood.labels),
    }

    X_all, y_all, src_all = [], [], []

    for split, (X, y) in splits.items():
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        X_all.append(X)
        y_all.append(y)
        src_all += [split] * len(X)

    # Load synthetic spectra
    for model_name in model_names:
        model_dir = os.path.join(generated_root, model_name)
        if not os.path.exists(model_dir):
            print(f"Skipping missing model folder: {model_dir}")
            continue
        label_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npy")])
        for f in label_files:
            path = os.path.join(model_dir, f)
            label_id = int(f.split("_")[0])
            arr = safe_load_array(path, device=device)
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                arr = arr.squeeze(1)
            elif isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = np.vstack(arr)
            X_all.append(arr)
            y_all.append(np.full(arr.shape[0], label_id))
            src_all += [f"{model_name}_label{label_id}"] * arr.shape[0]
            print(f"Loaded {f} ({arr.shape}, dtype={arr.dtype})")

    # Concatenate all
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    src_all = np.array(src_all)

    print(f"Total samples: {len(X_all)}, feature dim: {X_all.shape[1]}")

    # Normalize globally
    X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min() + 1e-8)

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=60, learning_rate='auto', init='pca', random_state=42)
    X_embedded = tsne.fit_transform(X_all)

    # Map numeric label → text label
    label_names = np.array([
        label_convergence.get(str(int(lbl)), str(int(lbl))) for lbl in y_all
    ])

    # Save
    os.makedirs(tsne_dir, exist_ok=True)
    X_tsne, labels, label_names, sources = dirs
    np.save(X_tsne, X_embedded.astype(np.float32))
    np.save(labels, y_all.astype(np.int16))
    np.save(label_names, label_names)
    np.save(sources, src_all)
    print(f"Saved t-SNE arrays to {tsne_dir}/")

if __name__ == "__main__":

    model_names = [
        "cvae_MLP3_32",
        "cvae_CNN3_8_MxP",
        "cgan_MLP3_32_weighted",
        "cgan_CNN3_32_weighted",
        "dm_S",
        "dm_M",
        "dm_L",
        "dm_XL",
        "dm_deep",
    ]

    tsne_dir = "results/tsne"
    X_tsne = np.load(os.path.join(tsne_dir, "X_tsne.npy"))
    labels = np.load(os.path.join(tsne_dir, "labels.npy"))
    label_names = np.load(os.path.join(tsne_dir, "label_names.npy"), allow_pickle=True)
    sources = np.load(os.path.join(tsne_dir, "sources.npy"), allow_pickle=True)
    dirs = [X_tsne, labels, label_names, sources]

    compute_tsne(tsne_dir, dirs, model_names)
    
    LABEL_NAMES = [
        'Enterobacter_cloacae_complex',
        'Enterococcus_Faecium',
        'Escherichia_Coli',
        'Klebsiella_Pneumoniae',
        'Pseudomonas_Aeruginosa',
        'Staphylococcus_Aureus'
    ]
    LABEL_TO_COLOR = {lbl: plt.cm.Spectral(i / 5) for i, lbl in enumerate(LABEL_NAMES)}

    # Filter training samples
    train_mask = sources == "train"
    X_train = X_tsne[train_mask]
    labels_train = label_names[train_mask]

    # Loop over all models in model_names and plot train vs generated for each
    labels_train = np.array([str(l).strip() for l in labels_train])
    idx_to_name = {str(i): LABEL_NAMES[i] for i in range(len(LABEL_NAMES))}
    labels_train_named = np.array([idx_to_name.get(l, l) for l in labels_train])

    for model_name in model_names:
        generated_prefix = f"{model_name}_label"
        gen_mask = np.char.startswith(sources.astype(str), generated_prefix)
        X_gen = X_tsne[gen_mask]
        labels_gen = label_names[gen_mask]
        labels_gen = np.array([str(l).strip() for l in labels_gen])
        labels_gen_named = np.array([idx_to_name.get(l, l) for l in labels_gen])

        plt.figure(figsize=(14, 7))
        for lbl in LABEL_NAMES:
            train_idx = np.where(labels_train_named == lbl)[0]
            if len(train_idx) > 0:
                plt.scatter(
                    X_train[train_idx, 0], X_train[train_idx, 1],
                    color=LABEL_TO_COLOR[lbl], marker='o', alpha=0.6,
                    label=f"Train: {lbl}", s=30, edgecolor='k', linewidth=0.5
                )
            gen_idx = np.where(labels_gen_named == lbl)[0]
            if len(gen_idx) > 0:
                plt.scatter(
                    X_gen[gen_idx, 0], X_gen[gen_idx, 1],
                    color=LABEL_TO_COLOR[lbl], marker='x', alpha=0.8,
                    label=f"{model_name}: {lbl}", s=40
                )

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE: Train (circles) vs {model_name} Generated (crosses) by Label")
        handles, labels_plt = plt.gca().get_legend_handles_labels()
        from collections import OrderedDict
        by_label = OrderedDict(zip(labels_plt, handles))
        # Place legend fully outside the plot
        plt.legend(by_label.values(), by_label.keys(), fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plot_name = f"tsne_train_vs_{model_name}_by_label.png"
        plt.savefig(os.path.join(tsne_dir, plot_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {os.path.join(tsne_dir, plot_name)}")