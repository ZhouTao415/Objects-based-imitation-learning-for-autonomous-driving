import torch
from torch.utils.data import DataLoader
from imitationLearning.utils.utils import make_abs_path, load_config
from imitationLearning.utils.dataset_utils import load_datasets
from imitationLearning.trainers.il_behaviour_cloner import BehaviourCloner
from imitationLearning.utils.model_utils import load_model, get_criterion_optimizer

# Define paths
DATA_PATH = make_abs_path(__file__, "data")
CHECKPOINT_PATH = "output/best_model.pth"  # Define checkpoint path

def main():
    config = load_config("configs/model.yaml")
    data_root = DATA_PATH  
    print(f"Using data path: {data_root}")

    # Load datasets
    _, _, test_loader = load_datasets(data_root, config["batch_size"])

    # Load trained model
    model = load_model(config, CHECKPOINT_PATH)

    # Loss function & Optimizer (optional, only needed if testing with loss)
    criterion, _ = get_criterion_optimizer(model, config)

    # Initialize BehaviourCloner (without training)
    bc = BehaviourCloner(config, model, None, None, criterion, None)

    # Evaluate model
    test_loss = bc.evaluate(test_loader)

    # Convert to Python floats before printing
    test_loss = tuple(float(x) for x in test_loss)  

    print(f"Test Loss (MSE, RMSE, MAE): {test_loss}")


if __name__ == "__main__":
    main()