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


def collect_species(dataset_path, preprocess_pipeline, species_list, logger):

    logger.info("Preprocessing pipeline:")
    for i, step in enumerate(preprocess_pipeline.preprocessors, 1):
        args = step.__dict__
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        logger.info(f"  {i}. {step.__class__.__name__}({arg_str})")

    # Initialize the DRIAMS Manager
    manager = DRIAMS_Manager(dataset_path)
    # Initialize arrays
    spectra = []
    labels = []
    metas = []
    logger.info("################## PROCESSING SPECIES ##################:")
    logger.info("Bacteria species included:")
    for species in species_list:
        logger.info(f"  - {species[0]} {species[1].lower()}")

    # Create dataset for the species
    dataset = DRIAMS_Dataset(manager, genus_species=species_list, preprocess_pipeline=preprocess_pipeline)

    # Process main species
    logger.info("Processing species spectra...")
    for i in tqdm(range(len(dataset)), desc="Extracting species"):
        spectrum_obj, label, meta = dataset[i]
        spectra.append(spectrum_obj.intensity)
        labels.append(label)
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

    return data, label, meta


def main(name, preprocess_pipeline, species_list, change_names=None):
    
    # Set up logging
    log_file = os.path.join(os.path.dirname(__file__), f'pickle_creation_{name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger('report_pickle_logger')

    # Define dataset path and save path
    dataset_path = "/export/data_ml4ds/bacteria_id/relevant_datasets/DRIAMS_PROCESSED_DATABASE"
    name_pickle = f"DRIAMS_study_{name}"
    save_path = os.path.join(os.path.dirname(__file__), name_pickle + '.pkl')

    # Log session start
    logger.info("=" * 80)
    logger.info(f"PICKLE CREATION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"Pickle name: {name_pickle}")
    logger.info(f"Save path: {save_path}")
    logger.info(f"Dataset source: {dataset_path}")
    

    data, label, meta = collect_species(dataset_path, preprocess_pipeline, species_list, logger)

    if change_names:
        logger.info("Updating labels according to change_names mapping...")
        # Update labels according to change_names mapping
        label = np.array([change_names.get(l, l) for l in label])

    # Save to pickle
    logger.info(f"Saving pickle to: {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump({'data': data, 'label': label, 'meta': meta}, f)
    logger.info("Pickle saved successfully!")

    logger.info("=" * 80)
    logger.info("SESSION COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":

    name = "clostri" #"MALDIGen"

    # Species to include
    species_list = [
        # ("Klebsiella", "Pneumoniae"),
        # ("Enterococcus", "Faecium"),
        # ("Staphylococcus", "Aureus"),
        # ("Escherichia", "Coli"),
        # ("Pseudomonas", "Aeruginosa"),
        # ("Enterobacter", "Aerogenes"),
        # ("Staphylococcus", "Saprophyticus"),
        # ("Proteus", "Vulgaris"),
        # ("Enterobacter", "Cloacae"),
        # ("Enterobacter", "Hormaechei"),
        # ("Enterobacter", "Asburiae"),
        # ("Enterobacter", "Kobei"),
        # ("Enterobacter", "Ludwigii"),
        # ("Enterobacter", "Roggenkampii"),
        ("Clostridium", "Difficile"),
    ] 

    change_names = {
        # "Enterobacter_Cloacae": "Enterobacter_cloacae_complex",
        # "Enterobacter_Hormaechei": "Enterobacter_cloacae_complex",
        # "Enterobacter_Asburiae": "Enterobacter_cloacae_complex",
        # "Enterobacter_Kobei": "Enterobacter_cloacae_complex",
        # "Enterobacter_Ludwigii": "Enterobacter_cloacae_complex",
        # "Enterobacter_Roggenkampii": "Enterobacter_cloacae_complex",
    }

    # Define the preprocessing pipeline
    preprocess_pipeline = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                                Smoother(halfwindow=10),
                                                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                                StdThresholder(factor=1.0),
                                                Trimmer(min=2000, max=20000),
                                                Binner(start=2000, stop=20000, step=3),
                                                LogScaler(base=10))     
    
    main(name, preprocess_pipeline, species_list, change_names)

