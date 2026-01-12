# MALDIVAS: AI-driven Generation of MALDI-TOF MS Spectra

MALDIVAS hosts the code that accompanies the preprint **“AI-driven Generation of MALDI-TOF MS for Microbial Characterization”** (Schmidt-Santiago *et al.*, 2025). The project benchmarks conditional Variational Autoencoders (MALDIVAEs), conditional GANs (MALDIGANs) and denoising diffusion models (MALDIffusion) for synthesizing realistic MALDI-TOF spectra that can be used to train downstream microbial classifiers, rebalance rare species and explore domain shift effects.

## Preprint at a Glance

> **AI-driven Generation of MALDI-TOF MS for Microbial Characterization**  
> Lucía Schmidt-Santiago, David Rodríguez-Temporal, Carlos Sevilla-Salcedo, Vanessa Gómez-Verdejo  
> arXiv:2511.17611 [cs.LG], submitted 18 Nov 2025 – https://doi.org/10.48550/arXiv.2511.17611
>
> Deep generative models conditioned on species labels (MALDIVAE, MALDIGAN and MALDIffusion) produce spectra that match the fidelity and diversity of real MALDI-TOF acquisitions. MALDIVAE strikes the best balance between realism, stability and compute cost, while MALDIffusion achieves the highest fidelity at a higher training budget and MALDIGAN remains competitive with slightly higher variance. Training classifiers solely on synthetic spectra reaches the accuracy of real-data baselines, and augmenting minority species reduces imbalance and domain mismatch without degrading authenticity.

Please cite the work if you build upon this repository:

```
@article{SchmidtSantiago2025maldivas,
  title     = {AI-driven Generation of MALDI-TOF MS for Microbial Characterization},
  author    = {Schmidt-Santiago, Lucía and Rodríguez-Temporal, David and Sevilla-Salcedo, Carlos and Gómez-Verdejo, Vanessa},
  journal   = {arXiv preprint arXiv:2511.17611},
  year      = {2025},
  url       = {https://doi.org/10.48550/arXiv.2511.17611}
}
```

## Getting Started

1. **Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The project targets Python ≥3.10 and PyTorch 2.8 with CUDA 12.8. CPU-only runs also work, albeit slower.

2. **Datasets**
   - Place the anonymized study pickles under `pickles/` (`MARISMa_study*.pkl`, `DRIAMS_study*.pkl`, `RKI*.pkl`).  
   - If you have raw MALDI-TOF exports, convert them with the scripts in `pickles/` (e.g., `python pickles/create_pickles_MARISMa.py`). Creation logs are stored in the same folder.

3. **Experiment configuration**
   - Copy or adapt a YAML file from `configs/`.  
   - Each config defines data sources, network architecture, optimizer settings and locations where checkpoints and plots will be stored (defaults under `results/`).

4. **Run**
   - Use the scripts in `experiments/` to train, evaluate and generate spectra from the desired model family (VAE, GAN, conditional GAN, diffusion).

## Running Experiments

### Configuration anatomy

All configs follow the same idea:

```yaml
results_dir: results              # base directory for logs/checkpoints
pickle_marisma: pickles/MARISMa_study.pkl
pickle_driams: pickles/DRIAMS_study.pkl
batch_size: 128
epochs: 500
input_dim: 6000                   # length of 1-D spectra
latent_dim: 8                     # bottleneck (varies by model)
encoder: CNNEncoder1D             # MLPEncoder1D, CNNEncoder1D, CNNAttenEncoder
decoder: CNNDecoder1D             # matching decoder type
model: cVAE                       # one of VAE_Bernoulli, cVAE, GAN, etc.
max_pool: true
```

Additional conditional fields include `y_species_dim`, `y_embed_dim`, `y_amr_dim` (for label embeddings) and pretrained checkpoint paths written back by the training scripts.

### Entry points

| Script | Description | Typical config prefix |
| --- | --- | --- |
| `experiments/main_VAE.py` | Species-conditioned VAE baseline | `configs/vae_*.yaml` |
| `experiments/main_cVAE.py` | Conditional VAE with label embeddings and AMR conditioning | `configs/cvae_*.yaml` |
| `experiments/main_GAN.py` | Unconditional / conditional GAN benchmark | `configs/gan_*.yaml` |
| `experiments/main_cGAN.py` | Weighted conditional GAN training | `configs/cgan_*.yaml` |
| `experiments/main_DM.py` | MALDIffusion 1-D UNet training | `configs/dm_*.yaml` |
| `experiments/generation.py` | Batch generation of spectra from saved checkpoints | uses the config files above |

Every script accepts the flag `--config PATH/TO/config.yaml`. Some scripts share optional switches:

| Flag | Meaning |
| --- | --- |
| `--train` | Train from scratch and save `best_model_*` checkpoints under `results/<model_family>/<config_name>/`. Without `--train`, the script loads the checkpoint paths stored in the YAML (e.g. `pretrained_model`). |
| `--evaluation` | Run ELBO/performance metrics, create latent-space visualizations, reconstructions, and timing stats. |
| `--pike` | Compute PIKE reconstruction errors (`losses/PIKE_GPU.py`) and write CSV summaries under `results/<...>/pike/`. |
| `--generation` | Sample spectra after training/evaluation; number of spectra per label is set by `--n_generate` (default 500). |

Example: train and evaluate a MALDIVAE variant.

```bash
python experiments/main_VAE.py \
  --config configs/vae_CNN3_8_MxP.yaml \
  --train --evaluation --pike --generation \
  --n_generate 1000
```

The command will log metrics to `results/vae/vae_CNN3_8_MxP`, store the best checkpoint, create plots (TSNE, reconstructions) and update the YAML with metadata and checkpoint paths. Re-running without `--train` will skip training and only use the saved weights.

The diffusion pipeline exposes a few extra knobs (see `experiments/main_DM.py --help`):

```bash
python experiments/main_DM.py \
  --config configs/dm_M.yaml \
  --train --evaluation --n_generate 5000
```

Diffusion checkpoints are written inside `results/dm/<config_name>/checkpoints/context_model_XX.pth`. The helper `experiments/generation.py` can read those checkpoints to create large synthetic corpora per label (saved under `results/generated_spectra/<model>/<label>_<name>.npy`).

### Visualizations and downstream evaluation

- `visualization/visualization.py` and `visualization/tsne_all.py` provide TSNE projections of latent codes and real-vs-synthetic comparisons.
- `utils/test_utils.py` implements reconstruction metrics (PIKE, inference timing) and CSV writers.
- `exp_MLP/` contains scripts to train MLP classifiers or assess OOD behaviour on synthetic datasets (`classification_analysis.py`, `classification_evaluation.py`).

## Repository Overview

| Path | Purpose |
| --- | --- |
| `configs/` | Canonical YAML configs for every model variant (VAE, cVAE, GAN, cGAN, diffusion) used in the paper. |
| `experiments/` | Main experiment scripts (`main_*.py`), generation helpers and small utilities such as `tsne_all.py`. |
| `dataloader/` | Dataset loaders for MARISMa, DRIAMS and RKI cohorts (`*_Manager.py`, `data.py`, `get_data.py`, `SpectrumObject`). These modules build PyTorch datasets and split train/val/test/OOD splits. |
| `models/` | Torch implementations of VAE backbones, GAN generators/discriminators, diffusion UNets and shared encoder/decoder blocks. Semi-supervised variants live in `SS_*.py`. |
| `losses/` | Custom losses and evaluation metrics, notably the GPU PIKE implementation plus generation-performance helpers. |
| `utils/` | Training/evaluation utilities (`training_utils.py` for logging/running loops, `test_utils.py` for metrics, `conditional_utils.py` for embedding preparation, `preprocess.py` for spectrum transforms). |
| `visualization/` | Plotting helpers, TSNE scripts and sample figures. |
| `pickles/` | Serialized datasets and the scripts to build them from raw MALDI-TOF exports. |
| `results/` | Default output tree for logs, checkpoints, plots and metadata generated by the experiment scripts. |
| `_helpers/` | Convenience scripts (`run_all_vaes.sh`, etc.) to schedule sweeps plus `experiments.txt` with notes on finished runs. |
| `exp_AMR/` | AMR-specific preprocessing (`create_pickles_MARISMa_AMR.py`) and training utilities for antibiotic resistance studies. |
| `exp_Clostridium/` | Minimal configs/scripts to reproduce Clostridium-focused VAE experiments. |
| `exp_MLP/` | Downstream classifiers, minority-class analysis notebooks and evaluation helpers for real vs. synthetic training sets. |
| `stats/` | Aggregate statistics for the datasets (JSON summaries, per-species split tables). Useful for sanity checking label distributions. |
| `otros/` | One-off analysis scripts and cached TSNE projections used in the manuscript. |
| `VANESSA/` | Research notebooks, auxiliary diffusion utilities and context-model checkpoints contributed by Vanessa Gómez-Verdejo (used for experimentation but not part of the main training harness). |
| `venv/` | Local virtual environment (optional; not tracked in version control). |
| `requirements.txt` | Locked dependency list that matches the experiments reported in the preprint. |

## Tips & Troubleshooting

- **GPU memory**: Diffusion (`dm_*`) configs can be demanding. Reduce `batch_size`, `num_blocks` or `base_features` in the YAML if you hit OOM errors.
- **Checkpoint management**: VAE scripts save `best_model_<config>.pt`; GAN scripts save both generator and discriminator checkpoints; diffusion saves `context_model_<epoch>.pth` per checkpoint. Update the YAML entries (`pretrained_model`, `pretrained_generator`, etc.) when moving files.
- **PIKE acceleration**: `losses/PIKE_GPU.py` expects CUDA. Switch to CPU by editing the helper to avoid GPU-only ops if necessary.
- **Custom species sets**: `pickles/process_manager.py` exposes utilities to harmonize label names. After creating new pickles, update `y_species_dim` and label mappings in your config.

## License / Contributions

The repository is provided as-is for research. Please open issues or contact the authors if you plan to extend the work or if you encounter problems reproducing the experiments.
