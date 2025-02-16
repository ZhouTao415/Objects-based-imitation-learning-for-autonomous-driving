import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from imitationLearning.data_loader.data_loader import DrivingDataset, ref_collate_fn
from imitationLearning.models.transformer_rnn_model import TransformerRNNModel
from imitationLearning.trainers.il_behaviour_cloner import BehaviourCloner
from imitationLearning.utils.utils import make_abs_path, load_config

# Create absolute paths for key data directories
DATA_PATH = make_abs_path(__file__, "data")


def main():
    config = load_config("configs/config.yaml")
    data_root = DATA_PATH # 确保 data/objects 与 data/waypoints 路径正确
    print(data_root)

    # 加载数据集并划分训练/验证集（例如 80%/20%）
    full_dataset = DrivingDataset(data_root)
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=ref_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=ref_collate_fn)
    
    model = TransformerRNNModel(
        obj_dim=config["obj_dim"],
        lane_dim=config["lane_dim"],
        imu_dim=config["imu_dim"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        output_dim=2   # 输出二维坐标
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    bc = BehaviourCloner(config, model, train_loader, valid_loader, criterion, optimizer)
    bc.train()

if __name__ == "__main__":
    main()
