import torch
from torch.utils.data import DataLoader, Subset
from imitationLearning.data_loader.data_loader import DrivingDataset, ref_collate_fn
from imitationLearning.utils.utils import make_abs_path


def load_datasets(data_root, batch_size, shuffle_train=True):
    """
    Load DrivingDataset and split it into training/validation/test sets in chronological order.
    
    Args:
        data_root (str): Path to the data
        batch_size (int): Batch size for DataLoader
        shuffle_train (bool): Whether to shuffle the training data (recommended to set to False for sequential data)

    Returns:
        train_loader, valid_loader, test_loader (DataLoader): Data loaders for training, validation, and test sets
    """
    full_dataset = DrivingDataset(data_root)
    
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = int(0.1 * dataset_size)    # 10% for validation
    test_size = dataset_size - train_size - val_size  # 10% for testing

 
    # Split in chronological order
    train_dataset = Subset(full_dataset, range(0, train_size))  # Take the first 80% of the data
    valid_dataset = Subset(full_dataset, range(train_size, train_size + val_size))  # Take the middle 10%
    test_dataset = Subset(full_dataset, range(train_size + val_size, dataset_size))  # Take the last 10%
   
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=ref_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=ref_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=ref_collate_fn)

    return train_loader, valid_loader, test_loader
