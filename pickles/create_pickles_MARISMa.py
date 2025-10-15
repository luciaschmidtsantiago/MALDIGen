import os
import sys
from tqdm import tqdm
import numpy as np
import pickle
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.MARISMa_Manager import MARISMaManager, MARISMa
from utils.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, StdThresholder, LogScaler, MinMaxScaler

# Set up logging
log_file = os.path.join(os.path.dirname(__file__), 'pickle_creation.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger('report_pickle_logger')

dataset_path = f"/export/data_ml4ds/bacteria_id/MARISMa"
name_pickle = 'MARISMa_study'
save_path = os.path.join(os.path.dirname(__file__), name_pickle + '.pkl')

# Log session start
logger.info("=" * 80)
logger.info(f"PICKLE CREATION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 80)
logger.info(f"Pickle name: {name_pickle}")
logger.info(f"Save path: {save_path}")
logger.info(f"Dataset source: {dataset_path}")


# Define the preprocessing pipeline
preproc = 'log10'

preprocess_pipeline = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                            Smoother(halfwindow=10),
                                            BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                            StdThresholder(factor=1.0),
                                            Trimmer(min=2000, max=20000),
                                            Binner(start=2000, stop=20000, step=3),
                                            LogScaler(base=10))       

logger.info("Preprocessing pipeline:")
for i, step in enumerate(preprocess_pipeline.preprocessors, 1):
    args = step.__dict__
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    logger.info(f"  {i}. {step.__class__.__name__}({arg_str})")

# Initialize the MALDIManager
pickle_path = os.path.join(os.path.dirname(__file__), 'MARISMa_anonymized.pkl')
manager = MARISMaManager(dataset_path, presaved=True, pickle_path=pickle_path)

genus_species_list = [
    ("Klebsiella", "Pneumoniae"),
    ("Enterococcus", "Faecium"),
    ("Staphylococcus", "Aureus"),
    ("Escherichia", "Coli"),
    ("Pseudomonas", "Aeruginosa"),
]

# Add Enterobacter cloacae complex species
enterobacter_species_list = [
    ("Enterobacter", "Cloacae"),
    ("Enterobacter", "Hormaechei"),
    ("Enterobacter", "Asburiae"),
    ("Enterobacter", "Kobei"),
    ("Enterobacter", "Ludwigii"),
    ("Enterobacter", "Roggenkampii"),
]

logger.info("Bacteria species included:")
logger.info("Main species:")
for genus, species in genus_species_list:
    logger.info(f"  - {genus} {species.lower()}")

logger.info("Enterobacter cloacae complex (will be labeled as 'Enterobacter cloacae complex'):")
for genus, species in enterobacter_species_list:
    logger.info(f"  - {genus} {species.lower()}")

# Query the main species
logger.info("Querying main species data...")
species = manager.query_spectra_dict(genus_species=genus_species_list)

# Query Enterobacter species separately
logger.info("Querying Enterobacter species data...")
enterobacter_data = manager.query_spectra_dict(genus_species=enterobacter_species_list)

# Create datasets for both main species and Enterobacter species
logger.info("Creating datasets...")
species_dataset = MARISMa(species, preprocess_pipeline=preprocess_pipeline)
enterobacter_dataset = MARISMa(enterobacter_data, preprocess_pipeline=preprocess_pipeline)

# Initialize arrays
spectra = []
labels = []
metas = []

# Process main species
logger.info("Processing main species spectra...")
for i in tqdm(range(len(species_dataset)), desc="Extracting main species"):
    spectrum_obj, label, meta = species_dataset[i]  # __getitem__ is called here
    spectra.append(spectrum_obj.intensity)  # get the intensity array
    labels.append(label)
    metas.append(meta)

# Process Enterobacter species and relabel them as "Enterobacter cloacae complex"
logger.info("Processing Enterobacter species spectra...")
for i in tqdm(range(len(enterobacter_dataset)), desc="Extracting Enterobacter species"):
    spectrum_obj, label, meta = enterobacter_dataset[i]  # __getitem__ is called here
    spectra.append(spectrum_obj.intensity)  # get the intensity array
    labels.append("Enterobacter_cloacae_complex")  # Relabel all Enterobacter species
    metas.append(meta)

# Convert to numpy arrays
data = np.stack(spectra)   # shape: (N_samples, N_features)
label = np.array(labels)    # shape: (N_samples,)
meta = np.array(metas)       # shape: (N_samples,)

logger.info(f"Final data shape: {data.shape}")
logger.info(f"Final label shape: {label.shape}")
unique_classes = np.unique(label)
logger.info(f"Unique classes ({len(unique_classes)}): {list(unique_classes)}")

# Log class distribution
logger.info("Class distribution:")
for class_name in unique_classes:
    count = np.sum(label == class_name)
    logger.info(f"  - {class_name}: {count} samples")

# Save to pickle
logger.info(f"Saving pickle to: {save_path}")
with open(save_path, 'wb') as f:
    pickle.dump({'data': data, 'label': label, 'meta': meta}, f)
logger.info("Pickle saved successfully!")

logger.info("=" * 80)
logger.info("SESSION COMPLETED")
logger.info("=" * 80)
