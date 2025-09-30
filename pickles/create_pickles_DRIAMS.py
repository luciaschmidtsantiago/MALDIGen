import os
import sys
from tqdm import tqdm
import numpy as np
import pickle
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.DRIAMS_Manager import DRIAMS_Manager, DRIAMS_Dataset
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

dataset_path = "/export/data_ml4ds/bacteria_id/relevant_datasets/DRIAMS_PROCESSED_DATABASE"
name_pickle = 'DRIAMS_study'
save_path = os.path.join(os.path.dirname(__file__), name_pickle + '.pkl')

# Log session start
logger.info("=" * 80)
logger.info(f"PICKLE CREATION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 80)
logger.info(f"Pickle name: {name_pickle}")
logger.info(f"Save path: {save_path}")
logger.info(f"Dataset source: {dataset_path}")

# Define the preprocessing pipeline
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

# Species to include
species_list = [
    ("Klebsiella", "Pneumoniae"),
    ("Enterococcus", "Faecium"),
    ("Staphylococcus", "Aureus"),
    ("Escherichia", "Coli"),
    ("Pseudomonas", "Aeruginosa"),
]

# Enterobacter cloacae complex
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
for genus, species in species_list:
    logger.info(f"  - {genus} {species.lower()}")
logger.info("Enterobacter cloacae complex (will be labeled as 'Enterobacter_cloacae_complex'):")
for genus, species in enterobacter_species_list:
    logger.info(f"  - {genus} {species.lower()}")

# Initialize the DRIAMS Manager
manager = DRIAMS_Manager(dataset_path)

# Create datasets
logger.info("Creating datasets...")
main_dataset = DRIAMS_Dataset(manager, genus_species=species_list, preprocess_pipeline=preprocess_pipeline)
enterobacter_dataset = DRIAMS_Dataset(manager, genus_species=enterobacter_species_list, preprocess_pipeline=preprocess_pipeline)

# Initialize arrays
spectra = []
labels = []
metas = []

# Process main species
logger.info("Processing main species spectra...")
for i in tqdm(range(len(main_dataset)), desc="Extracting main species"):
    spectrum_obj, label, meta = main_dataset[i]
    spectra.append(spectrum_obj.intensity)
    labels.append(label)
    metas.append(meta)

# Process Enterobacter species and relabel them as "Enterobacter_cloacae_complex"
logger.info("Processing Enterobacter species spectra...")
for i in tqdm(range(len(enterobacter_dataset)), desc="Extracting Enterobacter species"):
    spectrum_obj, label, meta = enterobacter_dataset[i]
    spectra.append(spectrum_obj.intensity)
    labels.append("Enterobacter_cloacae_complex")
    metas.append(meta)

# Convert to numpy arrays
data = np.stack(spectra)
label = np.array(labels)
meta = np.array(metas)

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
