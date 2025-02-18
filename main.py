import torch
from torch.utils.data import DataLoader
from imitationLearning.utils.utils import make_abs_path, load_config
from imitationLearning.utils.dataset_utils import load_datasets
from imitationLearning.trainers.il_behaviour_cloner import BehaviourCloner
from imitationLearning.utils.model_utils import create_model, get_criterion_optimizer

# Define paths
DATA_PATH = make_abs_path(__file__, "data")

def main():
    config = load_config("configs/model.yaml")
    data_root = DATA_PATH  
    print(f"Using data path: {data_root}")

    # Load datasets
    train_loader, valid_loader, _ = load_datasets(data_root, config["batch_size"])

    # Initialize model
    model = create_model(config)

    # Loss function & Optimizer
    criterion, optimizer = get_criterion_optimizer(model, config)

    # Initialize training pipeline
    bc = BehaviourCloner(config, model, train_loader, valid_loader, criterion, optimizer)
    
    # Train the model
    bc.train()

if __name__ == "__main__":
    main()