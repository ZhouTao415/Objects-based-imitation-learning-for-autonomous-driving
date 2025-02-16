import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# 处理 ImportError
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from imitationLearning.data_loader.data_loader import DrivingDataset, ref_collate_fn
from imitationLearning.models.transformer_rnn_model import TransformerRNNModel
from imitationLearning.utils.utils import make_abs_path, load_config

WAYPOINTS_PATH = make_abs_path(__file__, "../data/waypoints")


def convert_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def visualize_model_output(model, dataloader, device):
    """
    读取数据，运行模型，并绘制所有样本的 Waypoints 轨迹在**同一张图**上。
    """
    model.to(device)
    model.eval()
    waypoints_predicted = []
    waypoints_ground_truth = []
    
    with torch.no_grad():
        for batch in dataloader:
            obj = batch['objects'].to(device)
            lanes = batch['lanes'].to(device)
            lane_mask = batch['lane_mask'].to(device)
            imu = batch['imu'].to(device)
            waypoints = batch['waypoints'].to(device)
            output = model(obj, lanes, lane_mask, imu)  # 输出 shape (B, 4, 2)
            waypoints_predicted.append(convert_tensor_to_numpy(output))
            waypoints_ground_truth.append(convert_tensor_to_numpy(waypoints))
    
    if len(waypoints_predicted) == 0 or len(waypoints_ground_truth) == 0:
        raise ValueError("No valid data was loaded for visualization!")

    waypoints_predicted_arr = np.concatenate(waypoints_predicted, axis=0)   # shape (N, 4, 2)
    waypoints_ground_truth_arr = np.concatenate(waypoints_ground_truth, axis=0)  # shape (N, 4, 2)

    # ======== 1. 在同一张图上绘制所有轨迹 ========
    plt.figure(figsize=(10, 8))
    
    for n in range(min(20, waypoints_predicted_arr.shape[0])):  # 限制最多绘制 20 条轨迹
        plt.plot(
            waypoints_predicted_arr[n, :, 0],
            waypoints_predicted_arr[n, :, 1],
            linestyle="--",
            color="g",
            alpha=0.7,  # 透明度
            label="Prediction" if n == 0 else "",
        )

        plt.plot(
            waypoints_ground_truth_arr[n, :, 0],
            waypoints_ground_truth_arr[n, :, 1],
            linestyle="-",
            color="b",
            alpha=0.7,  # 透明度
            label="Ground Truth" if n == 0 else "",
        )

    plt.title("Waypoints: Ground Truth vs. Prediction (All Samples)")
    plt.ylabel("Longitudinal Distance")
    plt.xlabel("Lateral Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ======== 2. 绘制累积轨迹 ========
    waypoints_ground_truth_arr = np.reshape(waypoints_ground_truth_arr, (-1, 2))
    waypoints_predicted_arr = np.reshape(waypoints_predicted_arr, (-1, 2))

    plt.figure(figsize=(10, 8))
    
    plt.plot(
        np.cumsum(waypoints_predicted_arr[:, 0]),
        np.cumsum(waypoints_predicted_arr[:, 1]),
        linestyle="--",
        color="g",
        label="Prediction",
    )

    plt.plot(
        np.cumsum(waypoints_ground_truth_arr[:, 0]),
        np.cumsum(waypoints_ground_truth_arr[:, 1]),
        linestyle="-",
        color="b",
        label="Ground Truth",
    )

    plt.title("Cumulative Trajectory Comparison")
    plt.ylabel("Longitudinal Distance")
    plt.xlabel("Lateral Distance")
    plt.legend()
    plt.grid(True)
    plt.show()


def load_model(config):
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
    
    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(state_dict)

    return model


def main():
    config = load_config("configs/visualization.yaml")
    device = config["device"]

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    dataset = DrivingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=ref_collate_fn)

    model = load_model(config)
    visualize_model_output(model, dataloader, device)


if __name__ == "__main__":
    main()
