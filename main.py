import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
import numpy as np
from imitationLearning.data_loader.data_loader import DrivingDataset, ref_collate_fn
from imitationLearning.models.transformer_rnn_model import TransformerRNNModel
from imitationLearning.trainers.il_behaviour_cloner import BehaviourCloner

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config("configs/config.yaml")
    
    data_root = "data"  # 根目录，确保 data/objects 和 data/waypoints 存在
    train_dataset = DrivingDataset(data_root)
    valid_dataset = DrivingDataset(data_root)  # 实际项目中请划分训练集与验证集
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=ref_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=ref_collate_fn)
    
    model = TransformerRNNModel(
        obj_dim=config["obj_dim"],
        lane_feat_dim=config["lane_dim"],
        imu_dim=config["imu_dim"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"]
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    bc = BehaviourCloner(config, model, train_loader, valid_loader, criterion, optimizer)
    bc.train()

if __name__ == "__main__":
    main()
