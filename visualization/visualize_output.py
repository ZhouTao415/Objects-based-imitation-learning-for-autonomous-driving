import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec

from imitationLearning.data_loader.data_loader import DrivingDataset, ref_collate_fn
from imitationLearning.models.transformer_rnn_model import TransformerRNNModel
from imitationLearning.utils.utils import make_abs_path, load_config

# WAYPOINTS_PATH = make_abs_path(__file__, "../data/waypoints")


def convert_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def visualize_model_output(model, dataloader, device, save_path=None):
    """
    读取数据，运行模型，并绘制所有样本的 Waypoints 轨迹在**同一张 2x2 的 subplot** 上。
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
    print("waypoints_predicted_arr shape:", waypoints_predicted_arr.shape)
    waypoints_ground_truth_arr = np.concatenate(waypoints_ground_truth, axis=0)  # shape (N, 4, 2)
    print("waypoints_ground_truth_arr shape:", waypoints_ground_truth_arr.shape)

    # ======== 创建 `GridSpec` 布局 ========
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])  # 下面行高度增加

    ax1 = fig.add_subplot(gs[0, 0])  # 左上角 Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])  # 右上角 Prediction
    ax4 = fig.add_subplot(gs[1, 1])  # 右下角 Cumulative

    # ======== 1. 左上角（ax1）：所有 Ground Truth 轨迹 ========
    start_idx = 120
    end_idx = min(140, waypoints_ground_truth_arr.shape[0])  # 避免超出索引范围
    for n in range(start_idx, end_idx): 
        ax1.plot(
            waypoints_ground_truth_arr[n, :, 0],
            waypoints_ground_truth_arr[n, :, 1],
            linestyle="-",
            color="b",
            alpha=0.7,
        )
    ax1.set_title("Ground Truth Trajectories")
    ax1.set_xlabel("Lateral Distance")
    ax1.set_ylabel("Longitudinal Distance")
    ax1.grid(True)

    # ======== 2. 右上角（ax2）：所有 Prediction 轨迹 ========
    start_idx = 120
    end_idx = min(140, waypoints_predicted_arr.shape[0])
    for n in range(start_idx, end_idx):   
        ax2.plot(
            waypoints_predicted_arr[n, :, 0],
            waypoints_predicted_arr[n, :, 1],
            linestyle="--",
            color="g",
            alpha=0.7,
        )
    ax2.set_title("Predicted Trajectories")
    ax2.set_xlabel("Lateral Distance")
    ax2.set_ylabel("Longitudinal Distance")
    ax2.grid(True)

    # ======== 3. 细分左下角（ax3）为 5 个小 `subplot` ========
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, 0])  # 1行5列
    small_axes = [fig.add_subplot(gs2[0, i]) for i in range(5)]  # 生成5个小子图

    # 让 `ax3` 也显示 `120~140` 的轨迹
    start_idx = 120
    end_idx = min(140, waypoints_ground_truth_arr.shape[0])
    selected_indices = np.linspace(start_idx, end_idx - 1, num=5, dtype=int)  # 取5个索引

    for i, (ax, idx) in enumerate(zip(small_axes, selected_indices)):  
        ax.plot(
            waypoints_ground_truth_arr[idx, :, 0],
            waypoints_ground_truth_arr[idx, :, 1],
            marker="o",
            linestyle="-",
            color="b",
            label="Ground Truth" if i == 0 else "",
        )
        ax.plot(
            waypoints_predicted_arr[idx, :, 0],
            waypoints_predicted_arr[idx, :, 1],
            marker="o",
            linestyle="--",
            color="g",
            label="Prediction" if i == 0 else "",
        )
        ax.set_title(f"S {idx}")
        ax.set_xlabel("Lat Dist")
        ax.set_ylabel("Long Dist")
        ax.grid(True)

    small_axes[0].legend()  # 只在第一个子图上显示 legend


    # ======== 4. 右下角（ax4）：累积轨迹（Cumulative Trajectory） ========
    waypoints_ground_truth_arr = np.reshape(waypoints_ground_truth_arr, (-1, 2))
    print("reshaped waypoints_ground_truth_arr shape:", waypoints_ground_truth_arr.shape)
    waypoints_predicted_arr = np.reshape(waypoints_predicted_arr, (-1, 2))
    print("reshaped waypoints_predicted_arr shape:", waypoints_predicted_arr.shape)

    ax4.plot(
        np.cumsum(waypoints_ground_truth_arr[:, 0]),
        np.cumsum(waypoints_ground_truth_arr[:, 1]),
        linestyle="-",
        color="b",
        label="Ground Truth",
    )
    ax4.plot(
        np.cumsum(waypoints_predicted_arr[:, 0]),
        np.cumsum(waypoints_predicted_arr[:, 1]),
        linestyle="--",
        color="g",
        label="Prediction",
    )
    ax4.set_title("Cumulative Trajectory")
    ax4.set_xlabel("Lateral Distance")
    ax4.set_ylabel("Longitudinal Distance")
    ax4.legend()
    ax4.grid(True)
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    # 调整整体布局
    plt.tight_layout()
    
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

    data_root = make_abs_path(__file__, "../data")
    dataset = DrivingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=ref_collate_fn)

    model = load_model(config)

    # 读取 `plots_path`，默认保存到当前目录
    plots_path = config.get("plots_path", ".")
    save_path = os.path.join(plots_path, "trajectory_visualization.png")

    visualize_model_output(model, dataloader, device, save_path)


if __name__ == "__main__":
    main()
