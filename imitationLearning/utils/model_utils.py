# utils/model_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from imitationLearning.models.transformer_rnn_model import TransformerRNNModel

def create_model(config):
    """Initialize and return the TransformerRNNModel."""
    model = TransformerRNNModel(
        obj_dim=config["obj_dim"],
        lane_dim=config["lane_dim"],
        imu_dim=config["imu_dim"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        output_dim=2
    )
    return model

def load_model(config, checkpoint_path):
    """Load a trained model from a checkpoint."""
    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"], weights_only=True))
    return model

def get_criterion_optimizer(model, config):
    """Return loss function and optimizer."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    return criterion, optimizer
