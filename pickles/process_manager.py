import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.MARISMa_Manager import MARISMaManager
from dataloader.RKI_Manager import RKIManager


def main(data):
    # Create pickles for datasets

    if data == "MARISMa":
        # MARISMa dataset
        dataset_path = f"/export/data_ml4ds/bacteria_id/MARISMa_anonymized"

        # Initialize the MARISMa manager
        manager = MARISMaManager(dataset_path)
        pickle_path = os.path.join(os.path.dirname(__file__), 'MARISMa_anonymized.pkl')
        manager.save_to_pickle(pickle_path)

    elif data == "RKI":
        # RKI dataset
        dataset_path = f"/export/data_ml4ds/bacteria_id/relevant_datasets/10.5281/RKI_ROOT"

        # Initialize the RKI manager
        manager = RKIManager(dataset_path)
        pickle_path = os.path.join(os.path.dirname(__file__), 'RKI.pkl')
        manager.save_to_pickle(pickle_path)

    else:
        raise ValueError(f"Unknown dataset: {data}")
    
if __name__ == "__main__":
    # main("MARISMa")
    main("RKI")
