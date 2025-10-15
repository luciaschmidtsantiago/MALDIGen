import os
import pickle
import random
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset

from dataloader.SpectrumObject import SpectrumObject
from utils.visualization_old import visualize_preprocessing

class MARISMaManager:

    def __init__(self, root_dir, presaved=False, pickle_path=None):
        """
        Initializes the MARISMaManager.
        - root_dir: Root path of the MALDIMarañon dataset.
        """
        self.root_dir = root_dir

        if not presaved:
            self.spectra_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
            self._load_structure()

        else:
            if pickle_path is None:
                raise ValueError("Pickle path must be provided if presaved is True.")
            if not os.path.exists(pickle_path):
                raise FileNotFoundError(f"Pickle file {pickle_path} does not exist.")
            # Load the spectra_dict from the pickle file
            self.spectra_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
            self.load_from_pickle(pickle_path)
        
        self.stats = self.compute_statistics()  # Precompute statistics

    def _load_structure(self):
        """
        Traverse the hierarchical structure and populate spectra_dict with:
        spectra_dict[year][genus][species][study] -> list of (fid_path, acqu_path)
        """
        total_folders = sum(
            1 for year_folder in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, year_folder))
            for genus_folder in os.listdir(os.path.join(self.root_dir, year_folder)) if os.path.isdir(os.path.join(self.root_dir, year_folder, genus_folder))
            for species_folder in os.listdir(os.path.join(self.root_dir, year_folder, genus_folder)) if os.path.isdir(os.path.join(self.root_dir, year_folder, genus_folder, species_folder))
            for study_folder in os.listdir(os.path.join(self.root_dir, year_folder, genus_folder, species_folder)) if os.path.isdir(os.path.join(self.root_dir, year_folder, genus_folder, species_folder, study_folder))
        )        
        print(f"Total folders to process: {total_folders}")

        with tqdm(total=total_folders, desc="Processing Dataset", unit="folder") as pbar:
            for year in os.listdir(self.root_dir):
                year_path = os.path.join(self.root_dir, year)
                if not os.path.isdir(year_path):
                    continue

                for genus in os.listdir(year_path):
                    genus_path = os.path.join(year_path, genus)
                    if not os.path.isdir(genus_path):
                        continue

                    for species in os.listdir(genus_path):
                        species_path = os.path.join(genus_path, species)
                        if not os.path.isdir(species_path):
                            continue

                        for study in os.listdir(species_path):
                            study_path = os.path.join(species_path, study)
                            if not os.path.isdir(study_path):
                                continue

                            for target in os.listdir(study_path):
                                target_path = os.path.join(study_path, target)
                                if not os.path.isdir(target_path):
                                    continue

                                for acq_folder in os.listdir(target_path):
                                    acq_path = os.path.join(target_path, acq_folder, "1SLin")
                                    if not os.path.isdir(acq_path):
                                        continue

                                    fid_file = os.path.join(acq_path, "fid")
                                    acqu_file = os.path.join(acq_path, "acqu")
                                    if os.path.exists(fid_file) and os.path.exists(acqu_file):
                                        study_name = '/'.join((study, target, acq_folder))
                                        self.spectra_dict[year][genus][species][study_name].append((fid_file))
                                    else:
                                        ValueError(f"Missing fid or acqu file in {acq_path}")

                            pbar.update(1)
    
    def compute_statistics(self):
        """
        Computes a statistics DataFrame:
        - Rows = species (Genus_Species)
        - Columns = (Year -> Unique, Total)
        """
        # Diccionario auxiliar
        stats = {}

        # Recorrer toda la estructura
        for year, genus_dict in self.spectra_dict.items():
            for genus, species_dict in genus_dict.items():
                for species, studies_dict in species_dict.items():
                    species_name = f"{genus}_{species}"

                    if species_name not in stats:
                        stats[species_name] = {}  # <--- ADD THIS to fix the KeyError

                    total_count = len(studies_dict)

                    studies = []
                    for study_name, _ in studies_dict.items():
                        studies.append(study_name.split('/')[0])

                    unique_count = len(set(studies))

                    stats[species_name][(year, 'Unique')] = unique_count
                    stats[species_name][(year, 'Total')] = total_count

        # Crear DataFrame
        self.stats = pd.DataFrame.from_dict(stats, orient='index')

        # Ordenar columnas: primero por año y luego Unique/Total
        self.stats = self.stats.sort_index(axis=1, level=[0, 1])

        # Añadir nombre del índice
        self.stats.index.name = 'Species'

        return self.stats
    
    def save_to_pickle(self, file_path):
        """
        Saves the manager object (including spectra_dict and stats) to a pickle file.
        """
        # Convertir defaultdict a dict normal (sin lambdas)
        spectra_dict_clean = self._defaultdict_to_dict(self.spectra_dict)

        with open(file_path, 'wb') as f:
            pickle.dump({'spectra_dict': spectra_dict_clean}, f)

        print(f"✅ Manager saved to {file_path}")

    def _defaultdict_to_dict(self, d):
        """
        Recursively convert defaultdicts into normal dicts.
        """
        if isinstance(d, defaultdict):
            d = {k: self._defaultdict_to_dict(v) for k, v in d.items()}
        return d

    def load_from_pickle(self, file_path):
        """
        Loads the manager object (spectra_dict and stats) from a pickle file.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.spectra_dict = data['spectra_dict']
        print(f"✅ Manager loaded from {file_path}")

    def query_spectra_dict(self, years=None, genus=None, species=None, genus_species=None):
        """
        Query the spectra_dict with optional filters.

        Parameters:
            years (str or list of str or None): Filter by year(s). None = all years.
            genus (str or list of str or None): Filter by genus name(s). None = all genera.
            species (str or list of str or None): Filter by species name(s). None = all species.
            genus_species (list of (genus, species) tuples): Filter by genus-species pairs.

        Returns:
            filtered_dict: A filtered defaultdict with the same structure as self.spectra_dict.
        """
        # Normalize inputs to lists
        if isinstance(years, str):
            years = [years]
        if isinstance(genus, str):
            genus = [genus]
        if isinstance(species, str):
            species = [species]
        if genus_species is not None:
            genus_species = [(g.lower(), s.lower()) for g, s in genus_species]

        # Use all available years if None
        if years is None:
            years = list(self.spectra_dict.keys())

        filtered_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for year in years:
            if year not in self.spectra_dict:
                continue
            for g in self.spectra_dict[year]:
                for s in self.spectra_dict[year][g]:
                    # Filter by genus_species if provided
                    if genus_species is not None:
                        if (g.lower(), s.lower()) not in genus_species:
                            continue
                    else:
                        if genus and g.lower() not in [x.lower() for x in genus]:
                            continue
                        if species and s.lower() not in [x.lower() for x in species]:
                            continue
                    for study, fids in self.spectra_dict[year][g][s].items():
                        filtered_dict[year][g][s][study] = fids

        return filtered_dict
    
    def get_top_species(self, top_n=5):
        total_col = self.stats.columns.get_level_values(1) == "Total"
        totals = self.stats.loc[:, total_col].sum(axis=1)
        top_species = totals.sort_values(ascending=False).head(top_n).index.tolist()
        return [species.split('_') for species in top_species]  # [['Escherichia', 'Coli'], ...]


class MARISMa(Dataset):
    def __init__(self, spectra_dict, preprocess_pipeline=None, visualize=False, path=None):
        self.samples = []
        self.preprocess_pipeline = preprocess_pipeline

        for year, genus_dict in spectra_dict.items():
            for genus, species_dict in genus_dict.items():
                for species, studies in species_dict.items():
                    for study_name, fid_paths in studies.items():
                        for fid_path in fid_paths:
                            self.samples.append({
                                'fid': fid_path,
                                'label': f"{genus}_{species}",
                                'meta': {
                                    'year': year,
                                    'genus': genus,
                                    'species': species,
                                    'study': study_name
                                }
                            })

        if visualize and len(self.samples) > 0 and path:
            random_sample = random.choice(self.samples)
            fid_path = random_sample['fid']
            acqu_path = fid_path.replace('/fid', '/acqu')  # assuming standard Bruker structure
            spectrum = SpectrumObject.from_bruker(acqu_path, fid_path)
            visualize_preprocessing((spectrum, random_sample['meta']['study']), preprocess_pipeline, path)
        elif visualize and not path:
            raise ValueError("Path must be provided for visualization.")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        fid_path = entry['fid']
        acqu_path = fid_path.replace('/fid', '/acqu')  # assuming standard Bruker structure

        # Load and preprocess spectrum
        spectrum = SpectrumObject.from_bruker(acqu_path, fid_path)
        if self.preprocess_pipeline:
            spectrum = self.preprocess_pipeline(spectrum)

        # Return SpectrumObject and label
        return SpectrumObject(mz=spectrum.mz, intensity=spectrum.intensity), entry['label'], entry['meta']
