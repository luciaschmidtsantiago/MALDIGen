
# --- Fully redone script for t-SNE of all real and synthetic spectra ---
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.data import load_data
from utils.plotting_utils import LABEL_TO_HEX

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

def compute_tsne(model_names):

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

    # Normalize each real spectrum independently to [0, 1]
    for split, (X, y) in splits.items():
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        # Normalize each spectrum independently
        X_norm = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8)
        X_all.append(X_norm)
        y_all.append(y)
        src_all += [split] * len(X)

    # --- Diagnostic plot: visualize some training and generated spectra ---
    # Plot first 5 training spectra
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(X_all[0][i], label=f"Train {i}")
    plt.title("First 5 Training Spectra (Normalized)")
    plt.xlabel("Feature Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    save_dir = os.path.join("results", "tsne", "pruebas")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "first_5_training_spectra.png"), dpi=300)
    plt.show()

    # Plot first 5 generated spectra from first available model
    for model_name in model_names:
        model_dir = os.path.join(generated_root, model_name)
        if not os.path.exists(model_dir):
            continue
        label_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npy")])
        if label_files:
            arr = safe_load_array(os.path.join(model_dir, label_files[0]), device=device)
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                arr = arr.squeeze(1)
            arr_norm = (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True) + 1e-8)
            plt.figure(figsize=(12, 6))
            for i in range(min(5, arr_norm.shape[0])):
                plt.plot(arr_norm[i], label=f"{model_name} Gen {i}")
            plt.title(f"First 5 Generated Spectra ({model_name}, Normalized)")
            plt.xlabel("Feature Index")
            plt.ylabel("Intensity")
            plt.legend()
            plt.tight_layout()
            save_dir = os.path.join("results", "tsne", "pruebas")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"first_5_generated_spectra_{model_name}.png"), dpi=300)
            plt.show()
            break

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
            # Normalize each generated spectrum independently
            arr_norm = (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True) + 1e-8)
            X_all.append(arr_norm)
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

    return X_embedded, y_all, label_names, src_all

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

    printed_names = {
        'Enterobacter_cloacae_complex': 'E.cloacae complex',
        'Enterococcus_Faecium': 'E.faecium',
        'Escherichia_Coli': 'E.coli',
        'Klebsiella_Pneumoniae': 'K.pneumoniae',
        'Pseudomonas_Aeruginosa': 'P.aeruginosa',
        'Staphylococcus_Aureus': 'S.aureus',
        "Enterobacter_Aerogenes": 'E.aerogenes',
        "Staphylococcus_Saprophyticus" : 'S.saprophyticus',
        "Proteus_Vulgaris": 'P.vulgaris',
    }  

    tsne_dir = "results/tsne"
    X_tsne_path = os.path.join(tsne_dir, "X_tsne.npy")
    labels_path = os.path.join(tsne_dir, "labels.npy")
    label_names_path = os.path.join(tsne_dir, "label_names.npy")
    sources_path = os.path.join(tsne_dir, "sources.npy")

    compute = False
    if compute:
        X_embedded, y_all, label_names_arr, src_all = compute_tsne(model_names)

        # Save
        os.makedirs(tsne_dir, exist_ok=True)
        np.save(X_tsne_path, X_embedded.astype(np.float32))
        np.save(labels_path, y_all.astype(np.int16))
        np.save(label_names_path, label_names_arr)
        np.save(sources_path, src_all)
        print(f"Saved t-SNE arrays to {tsne_dir}/")

    # Load arrays for plotting
    X_tsne = np.load(X_tsne_path)
    labels = np.load(labels_path)
    label_names = np.load(label_names_path, allow_pickle=True)
    sources = np.load(sources_path, allow_pickle=True)
    
    # Hex colors from Spectral colormap
    LABEL_NAMES = list(LABEL_TO_HEX.keys())

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
            lbl_str = str(lbl)
            pretty = printed_names.get(lbl_str, lbl_str)
            pretty_italic = f"$\\it{{{pretty}}}$"
            train_idx = np.where(labels_train_named == lbl)[0]
            if len(train_idx) > 0:
                plt.scatter(
                    X_train[train_idx, 0], X_train[train_idx, 1],
                    color=LABEL_TO_HEX.get(lbl, '#888888'), marker='o', alpha=0.6,
                    label=f"Train: {pretty_italic}", s=30, edgecolor='k', linewidth=0.5
                )
            gen_idx = np.where(labels_gen_named == lbl)[0]
            if len(gen_idx) > 0:
                plt.scatter(
                    X_gen[gen_idx, 0], X_gen[gen_idx, 1],
                    color=LABEL_TO_HEX.get(lbl, '#888888'), marker='x', alpha=0.8,
                    label=f"Generated: {pretty_italic}", s=40
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



    # Plot t-SNE of train vs test samples
    test_mask = sources == "test"
    X_test = X_tsne[test_mask]
    labels_test = label_names[test_mask]
    labels_test = np.array([str(l).strip() for l in labels_test])
    labels_test_named = np.array([idx_to_name.get(l, l) for l in labels_test])

    plt.figure(figsize=(14, 7))
    for lbl in LABEL_NAMES:
        lbl_str = str(lbl)
        pretty = printed_names.get(lbl_str, lbl_str)
        pretty_italic = f"$\\it{{{pretty}}}$"
        train_idx = np.where(labels_train_named == lbl)[0]
        if len(train_idx) > 0:
            plt.scatter(
                X_train[train_idx, 0], X_train[train_idx, 1],
                color=LABEL_TO_HEX.get(lbl, '#888888'), marker='o', alpha=0.6,
                label=f"Train: {pretty_italic}", s=30, edgecolor='k', linewidth=0.5
            )
        test_idx = np.where(labels_test_named == lbl)[0]
        if len(test_idx) > 0:
            plt.scatter(
                X_test[test_idx, 0], X_test[test_idx, 1],
                color=LABEL_TO_HEX.get(lbl, '#888888'), marker='^', alpha=0.8,
                label=f"test: {pretty_italic}", s=40
            )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE: Train (circles) vs test (triangles) by Label")
    handles, labels_plt = plt.gca().get_legend_handles_labels()
    # Deduplicate legend entries while preserving mathtext formatting
    legend_items = []
    seen = set()
    for h, l in zip(handles, labels_plt):
        if l not in seen:
            legend_items.append((h, l))
            seen.add(l)
    handles_dedup, labels_dedup = zip(*legend_items)
    plt.legend(handles_dedup, labels_dedup, fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plot_name = "tsne_train_vs_test_by_label.png"
    plt.savefig(os.path.join(tsne_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {os.path.join(tsne_dir, plot_name)}")

    # Plot t-SNE of train vs OOD samples
    ood_mask = sources == "ood"
    X_ood = X_tsne[ood_mask]
    labels_ood = label_names[ood_mask]
    labels_ood = np.array([str(l).strip() for l in labels_ood])
    labels_ood_named = np.array([idx_to_name.get(l, l) for l in labels_ood])

    plt.figure(figsize=(14, 7))
    for lbl in LABEL_NAMES:
        lbl_str = str(lbl)
        pretty = printed_names.get(lbl_str, lbl_str)
        pretty_italic = f"$\\it{{{pretty}}}$"
        train_idx = np.where(labels_train_named == lbl)[0]
        if len(train_idx) > 0:
            plt.scatter(
                X_train[train_idx, 0], X_train[train_idx, 1],
                color=LABEL_TO_HEX.get(lbl, '#888888'), marker='o', alpha=0.6,
                label=f"Train: {pretty_italic}", s=30, edgecolor='k', linewidth=0.5
            )
        ood_idx = np.where(labels_ood_named == lbl)[0]
        if len(ood_idx) > 0:
            plt.scatter(
                X_ood[ood_idx, 0], X_ood[ood_idx, 1],
                color=LABEL_TO_HEX.get(lbl, '#888888'), marker='^', alpha=0.8,
                label=f"OOD: {pretty_italic}", s=40
            )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE: Train (circles) vs OOD (triangles) by Label")
    handles, labels_plt = plt.gca().get_legend_handles_labels()
    # Deduplicate legend entries while preserving mathtext formatting
    legend_items = []
    seen = set()
    for h, l in zip(handles, labels_plt):
        if l not in seen:
            legend_items.append((h, l))
            seen.add(l)
    handles_dedup, labels_dedup = zip(*legend_items)
    plt.legend(handles_dedup, labels_dedup, fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plot_name = "tsne_train_vs_ood_by_label.png"
    plt.savefig(os.path.join(tsne_dir, plot_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {os.path.join(tsne_dir, plot_name)}")