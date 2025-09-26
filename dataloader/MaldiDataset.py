import random
from torch.utils.data import Dataset

from dataloader.SpectrumObject import SpectrumObject
from utils.visualization_old import visualize_preprocessing

class MaldiDataset(Dataset):
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


class SynthDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        entry = self.samples[idx]
        spectrum = entry[0]
        label = entry[1]
        meta = entry[2]

        return spectrum, label, meta