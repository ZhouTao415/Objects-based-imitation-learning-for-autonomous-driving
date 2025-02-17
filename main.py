import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from imitationLearning.data_loader.data_loader import DrivingDataset, ref_collate_fn
from imitationLearning.models.transformer_rnn_model import TransformerRNNModel
from imitationLearning.trainers.il_behaviour_cloner import BehaviourCloner
from imitationLearning.utils.utils import make_abs_path, load_config

# Create absolute paths for key data directories
DATA_PATH = make_abs_path(__file__, "data")


def main():
    config = load_config("configs/model.yaml")
    data_root = DATA_PATH # 确保 data/objects 与 data/waypoints 路径正确
    print(data_root)

    # 加载数据集并划分训练/验证集（例如 80%/20%）
    full_dataset = DrivingDataset(data_root)
    
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)  # 80% 训练集
    val_size = int(0.1 * dataset_size)    # 10% 验证集
    test_size = dataset_size - train_size - val_size  # 10% 测试集
    
    # 2.1 random 划分
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    # 2.2 按时间顺序划分
    train_dataset = Subset(full_dataset, range(0, train_size))  # 取前 80% 数据
    valid_dataset = Subset(full_dataset, range(train_size, train_size + val_size))  # 取中间 10%
    test_dataset = Subset(full_dataset, range(train_size + val_size, dataset_size))  # 取最后 10%
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=ref_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=ref_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=ref_collate_fn)
    
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
    
    test_loss = bc.evaluate(test_loader)  # 假设 `evaluate()` 方法计算测试误差
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
