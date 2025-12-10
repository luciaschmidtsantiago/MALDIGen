import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.data import load_data

# ============================================================
# ================ MODEL GENERATION HELPER ===================
# ============================================================
def generate_samples(model, model_type, device, label_id, num_samples, config):
    """
    Generate synthetic spectra for one label using the correct method for each model type.
    """
    model.eval()
    model.to(device)

    if model_type == "vae":
        y_species = torch.full((num_samples,), label_id, dtype=torch.long, device=device)
        generated = model.sample(y_species)
        return generated

    elif model_type == "gan":
        y_species = torch.full((num_samples,), label_id, dtype=torch.long, device=device)
        z = torch.randn(num_samples, config.get('latent_dim', 32), device=device)
        with torch.no_grad():
            generated = model.forward_G(z, y_species)
        return generated

    elif model_type == "dm":
        timesteps = config.get('timesteps', 500)
        beta1 = config.get('beta1', 1e-4)
        beta2 = config.get('beta2', 0.02)
        b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device, dtype=torch.float32) + beta1
        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()
        ab_t[0] = 1.0

        # --- Batch-wise generation ---
        batch_size = min(500, num_samples)  # Adjust as needed for your GPU
        all_spectra = []
        context_dim = config.get('n_cfeat', config.get('n_classes', 6))
        L = model.length

        for start in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - start)
            c = torch.zeros(current_batch, context_dim, device=device)
            c[:, label_id] = 1.0
            x = torch.randn(current_batch, model.in_channels, L, device=device)

            with torch.no_grad():
                for t_inv in tqdm(range(timesteps, 0, -1), leave=False):
                    t = torch.full((current_batch,), t_inv, device=device, dtype=torch.long)
                    t_norm = (t.float() / float(timesteps)).view(-1, 1)
                    eps = model(x, t_norm, c)
                    ab = ab_t[t].view(current_batch, 1, 1)
                    a = a_t[t].view(current_batch, 1, 1)
                    b = b_t[t].view(current_batch, 1, 1)
                    x = (x - (b / (1 - ab).sqrt()) * eps) / a.sqrt()
                    if t_inv > 1:
                        x += b.sqrt() * torch.randn_like(x)

            from experiments.main_DM import denormalize_spectra
            spectra = denormalize_spectra(x).squeeze(1)
            all_spectra.append(spectra.detach().cpu())
            torch.cuda.empty_cache()

        return torch.cat(all_spectra, dim=0)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# ================ MODEL LOADING HELPER ======================
# ============================================================
def load_trained_model(model_name, model_path, config_path, device):
    """
    Load a trained model checkpoint given its name, result path, and config file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found for {model_name}: {config_path}")
    config = yaml.safe_load(open(config_path))

    # VAE
    if "vae" in model_name.lower():
        # Model architecture parameters
        D = config['input_dim']
        M = config['latent_dim']
        embedding = config.get('embedding', True)
        y_species_dim = config.get('y_species_dim', 0)
        y_embed_dim = config.get('y_embed_dim', 0)
        y_amr_dim = config.get('y_amr_dim', 0)
        cond_dim = y_embed_dim + y_amr_dim if embedding else y_species_dim + y_amr_dim
        num_layers = config['n_layers']
        max_pool = config.get('max_pool', False)
        
        from models.Networks import MLPEncoder1D, CNNEncoder1D, MLPDecoder1D, CNNDecoder1D
        encoder_type = config['encoder']
        decoder_type = config['decoder']
        encoder = MLPEncoder1D(D, num_layers, M, cond_dim=cond_dim).to(device) if encoder_type == 'MLPEncoder1D' else CNNEncoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool, cond_dim=cond_dim).to(device) if encoder_type == 'CNNEncoder1D' else None
        decoder = MLPDecoder1D(M, num_layers, D, cond_dim=cond_dim).to(device) if decoder_type == 'MLPDecoder1D' else CNNDecoder1D(M, (1, D), num_layers=num_layers, max_pool=max_pool, cond_dim=cond_dim).to(device) if decoder_type == 'CNNDecoder1D' else None
        
        from models.VAE import ConditionalVAE
        model = ConditionalVAE(encoder, decoder, y_species_dim, y_embed_dim, y_amr_dim, M, embedding).to(device)
        # Use config['pretrained_model'] if available, else search for best_model_*
        ckpt_path = config.get('pretrained_model', None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            # Search for best_model_* in model_path
            candidates = [f for f in os.listdir(model_path) if f.startswith('best_model_') and f.endswith('.pt')]
            if candidates:
                ckpt_path = os.path.join(model_path, sorted(candidates)[0])
            else:
                raise FileNotFoundError(f"VAE checkpoint not found for {model_name}: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        # Some VAE checkpoints may be dicts with 'model_state_dict'
        model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
        return model.to(device), config

    # GAN
    elif "gan" in model_name.lower():
        image_dim = config.get('output_dim', 6000)
        num_layers = config.get('n_layers', 3)
        batch_norm = config.get('batch_norm', False)
        latent_dim = config.get('latent_dim', 32)
        use_dropout = config.get('use_dropout', True)
        drop_p = config.get('dropout_prob', 0.1) if use_dropout else None
        y_species_dim = config.get('y_species_dim', 0)
        y_embed_dim = config.get('y_embed_dim', 0)
        y_amr_dim = config.get('y_amr_dim', 0)
        embedding = config.get('embedding', True)
        cond_dim = y_embed_dim + y_amr_dim if embedding else y_species_dim + y_amr_dim

        from models.GAN import MLPDecoder1D_Generator, CNNDecoder1D_Generator, Discriminator, ConditionalGAN
        gen_arch = config.get('generator', 'MLP')
        generator = MLPDecoder1D_Generator(latent_dim, num_layers, image_dim, cond_dim=cond_dim, use_bn=batch_norm).to(device) if gen_arch == 'MLP' else CNNDecoder1D_Generator(latent_dim, image_dim, n_layers=num_layers, cond_dim=cond_dim, use_dropout=use_dropout, dropout_prob=drop_p).to(device)
        discriminator = Discriminator(image_dim, cond_dim=cond_dim, use_bn=batch_norm, use_dropout=use_dropout, dropout_prob=drop_p).to(device)
        model = ConditionalGAN(generator, discriminator, y_species_dim, y_embed_dim, y_amr_dim, embedding).to(device)

        # Use config['pretrained_generator'] and config['pretrained_discriminator'] if available
        pretrained_generator = config.get('pretrained_generator', os.path.join(model_path, "best_generator.pt"))
        pretrained_discriminator = config.get('pretrained_discriminator', os.path.join(model_path, "best_discriminator.pt"))
        if pretrained_generator is None or pretrained_discriminator is None:
            raise ValueError("Pretrained model paths must be specified in the config for evaluation mode.")
        model.generator.load_state_dict(torch.load(pretrained_generator, map_location=device))
        model.discriminator.load_state_dict(torch.load(pretrained_discriminator, map_location=device))
        return model.to(device), config

    # Diffusion Model
    elif "dm" in model_name.lower():
        output_dim = config.get('output_dim', 6000)
        n_classes = config.get('n_classes', 6)
        timesteps = config.get('timesteps', 500)
        beta1 = config.get('beta1', 1e-4)
        beta2 = config.get('beta2', 0.02)
        n_channels = config.get('n_channels', 1)
        base_features = config.get('base_features', 64)
        num_blocks = config.get('num_blocks', 2)
        n_cfeat = config.get('n_cfeat', n_classes)
        norm_groups = config.get('norm_groups', 8)
        kernel_size = config.get('kernel_size', 4)
        b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device, dtype=torch.float32) + beta1
        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()
        ab_t[0] = 1.0

        from models.DM import ContextUnet1D
        model = ContextUnet1D(n_channels, base_features, n_cfeat, output_dim, num_blocks, norm_groups, kernel_size).to(device)

        # Use config['pretrained_model_path'] if available, else get last checkpoint from checkpoints dir
        pretrained_path = config.get('pretrained_model_path', None)
        if not pretrained_path or not os.path.exists(pretrained_path):
            # Find last checkpoint in checkpoints dir
            ckpt_dir = os.path.join(model_path, "checkpoints")
            if not os.path.exists(ckpt_dir):
                raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")
            candidates = [f for f in os.listdir(ckpt_dir) if f.startswith('context_model_') and f.endswith('.pth')]
            if not candidates:
                raise FileNotFoundError(f"No DM checkpoints found in {ckpt_dir}")
            # Sort by epoch number (assumes context_model_{epoch}.pth)
            candidates_sorted = sorted(candidates, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            pretrained_path = os.path.join(ckpt_dir, candidates_sorted[-1])
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        return model.to(device), config

    else:
        raise ValueError(f"Unknown model type for {model_name}")


# ============================================================
# ===================== MAIN SCRIPT ==========================
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Define models and output directories ---
    model_paths = {
        "cvae_MLP3_32": "results/vae/cvae_MLP3_32",
        "cvae_CNN3_8_MxP": "results/vae/cvae_CNN3_8_MxP",
        "cgan_MLP3_32_weighted": "results/gan/cgan_MLP3_32_weighted",
        "cgan_CNN3_32_weighted": "results/gan/cgan_CNN3_32_weighted",
        "dm_S": "results/dm/dm_S",
        "dm_M": "results/dm/dm_M",
        "dm_L": "results/dm/dm_L",
        "dm_XL": "results/dm/dm_XL",
        "dm_deep": "results/dm/dm_deep",
        "cvae_CNN3_8_MxP_extended": "results/vae/cvae_CNN3_8_MxP_extended",
        "cgan_CNN3_32_weighted_extended": "results/gan/cgan_CNN3_32_weighted_extended",
        "dm_deep_extended": "results/dm/dm_deep_CNN3_8_MxP_extended",
    }

    # --- Output base directory ---
    out_base = "results/generated_spectra"
    os.makedirs(out_base, exist_ok=True)

    # ============================================================
    # Generate per-model, per-label synthetic spectra
    # ============================================================
    for model_name, model_dir in model_paths.items():
        print(f"\n🔹 Generating synthetic spectra for model: {model_name}")
        out_dir = os.path.join(out_base, model_name)
        os.makedirs(out_dir, exist_ok=True)

        if 'extended' in model_name:
            pickle_marisma = "pickles/MARISMa_study_extended.pkl"
            pickle_driams = "pickles/DRIAMS_study_extended.pkl"
            num_samples = 5000
        elif 'enterobacter' in model_name:
            pickle_marisma = "pickles/MARISMa_study_enterobacter.pkl"
            pickle_driams = "pickles/DRIAMS_study_enterobacter.pkl"
            num_samples = 5000
        else:
            pickle_marisma = "pickles/MARISMa_study.pkl"
            pickle_driams = "pickles/DRIAMS_study.pkl"
            num_samples = 25000

        # Find config file (assume naming convention: configs/{model_name}.yaml)
        config_path = f"configs/{model_name}.yaml"
        if not os.path.exists(config_path):
            print(f"  ⚠️ Config file not found for {model_name}: {config_path}")
            continue
        model_type = "vae" if "vae" in model_name.lower() else "gan" if "gan" in model_name.lower() else "dm" if "dm" in model_name.lower() else None

        # Load trained model and config
        print(f"  Loading model and config for {model_name}...")
        model, config = load_trained_model(model_name, model_dir, config_path, device)
        print(f"  Model loaded successfully.")

        # Load label mapping (from training data) -- diffusion needs model_type
        print(f"  Loading label mapping...")
        if model_type == "dm":
            train, val, test, ood = load_data(pickle_marisma, pickle_driams, logger=None, get_labels=True, model_type="diffusion")
        else:
            train, val, test, ood = load_data(pickle_marisma, pickle_driams, logger=None, get_labels=True)
        label_convergence = train.label_convergence
        labels = sorted(label_convergence.keys(), key=lambda x: int(x))
        print(f"  Found {len(labels)} labels: {labels}")

        print(f"  Generating {num_samples} samples per label...")
        for lbl_str in labels:
            lbl = int(lbl_str)
            lbl_name = label_convergence[lbl_str]
            print(f"    → Generating for label {lbl_name} (id={lbl})...")
            spectra = generate_samples(model, model_type, device, lbl, num_samples, config)
            spectra = spectra.detach().cpu().numpy()
            out_path = os.path.join(out_dir, f"{lbl}_{lbl_name}.npy")
            np.save(out_path, spectra)
            print(f"      ✅ Saved {num_samples} samples for label {lbl_name} → {out_path}")

        print(f"  Finished generation for model: {model_name}")
        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()