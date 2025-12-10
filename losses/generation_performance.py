# --- Redone script: Compute PIKE matrices for generated vs train for each model ---
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.data import load_data, get_dataloaders
from PIKE_GPU import calculate_PIKE_gpu
from baseline_metric import get_fold, compute_mmd_from_pike, compute_class_distance_stats, compute_neighbour_distance_stats
from experiments.tsne_all import safe_load_array


def compute_pike_matrix(X_gen, X_train, t=8, device="cpu", label_id=None, model_name=None):
    """
    Compute PIKE matrix between all generated and all train samples.
    X_gen: [N_gen, D], X_train: [N_train, D]
    Returns: [N_gen, N_train] numpy array
    """
    X_gen = torch.from_numpy(X_gen).to(device)
    X_train = torch.from_numpy(X_train).to(device)
    N_gen, N_train = X_gen.shape[0], X_train.shape[0]
    K = np.zeros((N_gen, N_train), dtype=np.float32)
    print(f"      [PIKE] Computing PIKE matrix: {N_gen} gen x {N_train} train (label {label_id}, model {model_name})")
    with torch.no_grad():
        for i in tqdm(range(N_gen), desc=f"PIKE: {model_name} label {label_id}", leave=False):
            xi = X_gen[i].view(-1)
            for j in range(N_train):
                yj = X_train[j].view(-1)
                val = calculate_PIKE_gpu(xi, yj, t)
                if not np.isfinite(val):
                    val = 0.0
                K[i, j] = float(val)
    return K

def load_generated_spectra(X_train, y_train, label_convergence, device, gen_root, out_root):
    model_names = [d for d in os.listdir(gen_root) if os.path.isdir(os.path.join(gen_root, d))]
    model_names = ['dm_deep', 'cgan_CNN3_32_weighted','cvae_CNN3_8_MxP']

    print(f"[INFO] Found {len(model_names)} models in {gen_root}")
    for model_name in tqdm(model_names, desc="Models", position=0):
        model_dir = os.path.join(gen_root, model_name)
        out_dir = os.path.join(out_root, model_name)
        os.makedirs(out_dir, exist_ok=True)
        label_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npy")])
        print(f"\n[INFO] Processing model: {model_name} ({len(label_files)} label files)")
        for f in tqdm(label_files, desc=f"{model_name} labels", position=1, leave=False):
            label_id = f.split("_")[0]
            label_name = label_convergence.get(label_id, label_id)
            print(f"    [INFO] Label {label_id} ({label_name})")
            gen_path = os.path.join(model_dir, f)
            X_gen = safe_load_array(gen_path, device=device)
            if X_gen.ndim == 3 and X_gen.shape[1] == 1:
                X_gen = X_gen.squeeze(1)
            if X_gen.ndim == 1:
                X_gen = X_gen[None, :]
            # Remove zero spectra
            X_gen = X_gen[np.abs(X_gen).sum(axis=1) > 0]
            # Get all train samples for this label (from the fold)
            idx_train = np.where(y_train == int(label_id))[0]
            if len(idx_train) == 0 or X_gen.shape[0] == 0:
                print(f"      [WARN] No train or gen samples for label {label_id}, skipping.")
                continue
            X_train_label = X_train[idx_train]
            # Remove zero spectra
            X_train_label = X_train_label[np.abs(X_train_label).sum(axis=1) > 0]
            if X_train_label.shape[0] == 0:
                print(f"      [WARN] No nonzero train samples for label {label_id}, skipping.")
                continue
            # Compute PIKE matrix
            K = compute_pike_matrix(X_gen, X_train_label, t=8, device=device, label_id=label_id, model_name=model_name)
            out_path = os.path.join(out_dir, f"pike_gen_vs_train_label{label_id}.npy")
            np.save(out_path, K)
            print(f"      [OK] Saved PIKE matrix: {out_path}")

def compute_all_generation_metrics(gen_root, gen_train_root, gen_gen_root, train_train_root, label_convergence, model_name, device):
    """
    Compute all metrics (PIKE mean±std, MMD², class distance, neighbour distance)
    for generated vs train PIKE matrices, and save as generation_metrics.csv.
    """
    # Setup: ensure mapping uses correct species names
    labels = sorted(np.unique(y_train).tolist())
    label_ids = [int(l) for l in labels]
    label_names = []
    for l in label_ids:
        if l in label_convergence:
            label_names.append(label_convergence[l])
        elif str(l) in label_convergence:
            label_names.append(label_convergence[str(l)])
        else:
            label_names.append(f"{l}")


    row_pike = []
    row_mmd = []
    row_dclass = []
    row_dnn = []

    for label_id, label_name in zip(label_ids, label_names):
        path_gen_train = os.path.join(gen_train_root, model_name, f"pike_gen_vs_train_label{label_id}.npy")
        path_train_train = os.path.join(train_train_root, f"pairwise_PIKE_label{label_id}.npy")
        path_gen_gen = os.path.join(gen_gen_root, model_name, f"pike_gen_vs_gen_label{label_id}.npy")
        if not os.path.exists(path_gen_gen):
            path_gen_gen = os.path.join(train_train_root, f"pairwise_PIKE_label{label_id}.npy")

        K_gen_train = np.load(path_gen_train)
        K_gen_train = np.nan_to_num(K_gen_train, nan=0.0, posinf=1.0, neginf=0.0)
        K_train_train = np.load(path_train_train)
        K_train_train = np.nan_to_num(K_train_train, nan=0.0, posinf=1.0, neginf=0.0)
        K_gen_gen = np.load(path_gen_gen)
        K_gen_gen = np.nan_to_num(K_gen_gen, nan=0.0, posinf=1.0, neginf=0.0)

        # PIKE mean±std (gen vs train)
        mean_pike = float(np.mean(K_gen_train))
        std_pike = float(np.std(K_gen_train))
        row_pike.append(f"{mean_pike:.3f}±{std_pike:.3f}")

        # MMD² calculation (unbiased) using compute_mmd_from_pike
        mmd2 = compute_mmd_from_pike(K_train_train, K_gen_gen, K_gen_train)

        # Other metrics (on gen vs train)
        mean_dclass, std_dclass = compute_class_distance_stats(K_gen_train)
        mean_dnn, std_dnn = compute_neighbour_distance_stats(K_gen_train)

        print(f"Label {label_name}: MMD²={mmd2:.6f}, D_class={mean_dclass:.6f}±{std_dclass:.6f}, D_NN={mean_dnn:.6f}±{std_dnn:.6f}")

        row_mmd.append(f"{mmd2:.6f}")
        row_dclass.append(f"{mean_dclass:.3f}±{std_dclass:.3f}")
        row_dnn.append(f"{mean_dnn:.3f}±{std_dnn:.3f}")

    # Write compact CSV table
    out_csv = os.path.join(gen_root, f"generation_metrics_{model_name}.csv")
    with open(out_csv, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["Label_ID"] + [str(i) for i in label_ids])
        writer.writerow(["Species_name"] + label_names)
        writer.writerow(["PIKE"] + row_pike)
        writer.writerow(["MMD²"] + row_mmd)
        writer.writerow(["Class_dist"] + row_dclass)
        writer.writerow(["Neighbour_dist"] + row_dnn)

    print(f"✅ Saved generation metrics table to {out_csv}")

def compute_pike_gen_vs_gen(gen_root, gen_gen_root, device):
    for model_name in ['dm_deep', 'cgan_CNN3_32_weighted', 'cvae_CNN3_8_MxP']:
        model_dir = os.path.join(gen_root, model_name)
        out_dir = os.path.join(gen_gen_root, model_name)
        os.makedirs(out_dir, exist_ok=True)
        label_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npy")])
        print(f"\n[INFO] PIKE gen_vs_gen for model: {model_name} ({len(label_files)} label files)")
        for f in label_files:
            label_id = f.split("_")[0]
            label_name = label_convergence.get(label_id, label_id)
            print(f"    [INFO] Label {label_id} ({label_name})")
            gen_path = os.path.join(model_dir, f)
            X_gen = safe_load_array(gen_path, device=device)
            if X_gen.ndim == 3 and X_gen.shape[1] == 1:
                X_gen = X_gen.squeeze(1)
            if X_gen.ndim == 1:
                X_gen = X_gen[None, :]
            # Remove zero spectra
            X_gen = X_gen[np.abs(X_gen).sum(axis=1) > 0]
            n = X_gen.shape[0]
            if n < 2:
                print(f"      [WARN] Not enough generated samples for label {label_id}, skipping.")
                continue
            print(f"      [INFO] Computing {n}x{n} PIKE gen_vs_gen matrix")
            X_gen_tensor = torch.from_numpy(X_gen).to(device)
            sims = torch.zeros((n, n), dtype=torch.float32, device=device)
            zero_mask = (X_gen_tensor.abs().sum(dim=1) == 0)
            with torch.no_grad():
                for i in tqdm(range(n), desc=f"label {label_id} rows", leave=False):
                    x_i = X_gen_tensor[i].view(-1)
                    if zero_mask[i]:
                        sims[i, :] = 0.0
                        continue
                    for j in range(n):
                        y_j = X_gen_tensor[j].view(-1)
                        if zero_mask[j]:
                            sims[i, j] = 0.0
                            continue
                        val = calculate_PIKE_gpu(x_i, y_j, 8)
                        if np.isnan(val) or np.isinf(val):
                            val = 0.0
                        sims[i, j] = float(val)
            # Diagonal = 1.0
            idx = torch.arange(n, device=device)
            sims[idx, idx] = 1.0
            np.save(os.path.join(out_dir, f"pike_gen_vs_gen_label{label_id}.npy"), sims.detach().cpu().numpy())
            print(f"      [OK] Saved PIKE gen_vs_gen matrix for label {label_id} ({n}x{n})")

if __name__ == "__main__":
    # For each model in results/generated_spectra
    gen_root = "results/generated_spectra"
    gen_train_root = "results/generated_spectra/pike_gen/pike_gen_vs_train"
    gen_gen_root = "results/generated_spectra/pike_gen/pike_gen_vs_gen"
    train_train_root = "results/baselines"
    os.makedirs(gen_train_root, exist_ok=True)
    os.makedirs(gen_gen_root, exist_ok=True)

    pickle_marisma = "pickles/MARISMa_study.pkl"
    pickle_driams = "pickles/DRIAMS_study.pkl"
    train, val, test, ood = load_data(pickle_marisma, pickle_driams, logger=None, get_labels=True)
    label_convergence = train.label_convergence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same stratified 10% training fold as baseline_metric.py
    train_loader, _, _, _ = get_dataloaders(train, val, test, ood, batch_size=64)
    X_subset, y_subset = get_fold(train_loader, subset_ratio=0.1, seed=42)
    X_train = X_subset.cpu().numpy() if hasattr(X_subset, 'cpu') else X_subset
    y_train = y_subset.cpu().numpy() if hasattr(y_subset, 'cpu') else y_subset

    # COMPUTE PAIRWISE PIKE PER LABEL (GENERATED VS TRAINING)
    load_generated_spectra(X_train, y_train, label_convergence, device, gen_root, gen_train_root)

    # COMPUTE PIKE OF GENERATED VS GENERATED
    compute_pike_gen_vs_gen(gen_root, gen_gen_root, device)

    # Compute metrics for each model
    for model_name in ['dm_deep', 'cgan_CNN3_32_weighted', 'cvae_CNN3_8_MxP']:
        compute_all_generation_metrics(gen_root, gen_train_root, gen_gen_root, train_train_root, label_convergence, model_name, device)

