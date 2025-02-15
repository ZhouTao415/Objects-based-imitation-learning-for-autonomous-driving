import os 

import numpy as np
import torch
from torch.utils.data import DataLoader
from imitationLearning.data_loader.data_lodaer import DrivingDataset

# from data_loader.data_loader import DrivingDataset
from imitationLearning.utils.utils import make_abs_path


WAYPOINTS_PATH = make_abs_path(__file__, "../data/waypoints")
OBJECTS_PATH = make_abs_path(__file__, "../data/objects")

def test_dataset_dir():
    """
    Check if the `waypoints/` and `objects/` directories in the dataset match.
    - If the path does not exist, raise `FileNotFoundError`
    - If the contents of the directories do not match, print the specific missing items
    """

    waypoints_contents = set(os.listdir(WAYPOINTS_PATH))
    objects_contents = set(os.listdir(OBJECTS_PATH))

    assert waypoints_contents == objects_contents, (
        f"Dataset directories do not match!\n"
    )

    print("✅ Dataset directory check passed, waypoints and objects folders match.")


if __name__ == "__main__":
    test_dataset_dir()
