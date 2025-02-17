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

DATA_PATH = make_abs_path(__file__, "../../../data")

def convert_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def plot_ground_truth(ax, ground_truth_arr, start_idx=120, end_idx=140):
    """绘制左上角：所有 Ground Truth 轨迹"""
    end_idx = min(end_idx, ground_truth_arr.shape[0])
    for n in range(start_idx, end_idx):
        ax.plot(
            ground_truth_arr[n, :, 0],
            ground_truth_arr[n, :, 1],
            marker="o",
            linestyle="-",
            color="b",
            alpha=0.7,
        )
    ax.set_title("Ground Truth Trajectories")
    ax.set_xlabel("Lateral Distance")
    ax.set_ylabel("Longitudinal Distance")
    ax.grid(True)

def plot_prediction(ax, predicted_arr, start_idx=120, end_idx=140):
    """绘制右上角：所有 Prediction 轨迹"""
    end_idx = min(end_idx, predicted_arr.shape[0])
    for n in range(start_idx, end_idx):
        ax.plot(
            predicted_arr[n, :, 0],
            predicted_arr[n, :, 1],
            marker="o",
            linestyle="--",
            color="g",
            alpha=0.7,
        )
    ax.set_title("Predicted Trajectories")
    ax.set_xlabel("Lateral Distance")
    ax.set_ylabel("Longitudinal Distance")
    ax.grid(True)

def plot_subplots(axs, ground_truth_arr, predicted_arr, start_idx=120, end_idx=140, num_plots=5):
    """绘制左下角：5个小 subplot，分别展示单个样本的预测与真实轨迹对比"""
    end_idx = min(end_idx, ground_truth_arr.shape[0])
    selected_indices = np.linspace(start_idx, end_idx - 1, num=num_plots, dtype=int)
    for i, (ax, idx) in enumerate(zip(axs, selected_indices)):
        ax.plot(
            ground_truth_arr[idx, :, 0],
            ground_truth_arr[idx, :, 1],
            marker="o",
            linestyle="-",
            color="b",
            label="Ground Truth" if i == 0 else "",
        )
        ax.plot(
            predicted_arr[idx, :, 0],
            predicted_arr[idx, :, 1],
            marker="o",
            linestyle="--",
            color="g",
            label="Prediction" if i == 0 else "",
        )
        ax.set_title(f"Sample {idx}")
        ax.set_xlabel("Lat Dist")
        ax.set_ylabel("Long Dist")
        ax.grid(True)
    axs[0].legend()

def plot_cumulative(ax, ground_truth_arr, predicted_arr):
    """绘制右下角：累积轨迹对比"""
    gt = np.reshape(ground_truth_arr, (-1, 2))
    pred = np.reshape(predicted_arr, (-1, 2))
    ax.plot(
        np.cumsum(gt[:, 0]),
        np.cumsum(gt[:, 1]),
        linestyle="-",
        color="b",
        label="Ground Truth",
    )
    ax.plot(
        np.cumsum(pred[:, 0]),
        np.cumsum(pred[:, 1]),
        linestyle="--",
        color="g",
        label="Prediction",
    )
    ax.set_title("Cumulative Trajectory")
    ax.set_xlabel("Lateral Distance")
    ax.set_ylabel("Longitudinal Distance")
    ax.legend()
    ax.grid(True)

def visualize_model_output(model, dataloader, device, save_path=None):
    """
    读取数据、运行模型，并绘制所有样本的 Waypoints 轨迹，
    使用 2x2 的 subplot 布局展示各个角度的对比。
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

    # ======== 创建 GridSpec 布局 ========
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])  # 左上角：Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])  # 右上角：Prediction
    ax4 = fig.add_subplot(gs[1, 1])  # 右下角：Cumulative Trajectory

    # 调用单独函数绘制各个部分
    plot_ground_truth(ax1, waypoints_ground_truth_arr)
    plot_prediction(ax2, waypoints_predicted_arr)

    # 细分左下角为 5 个子图
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, 0])
    small_axes = [fig.add_subplot(gs2[0, i]) for i in range(5)]
    plot_subplots(small_axes, waypoints_ground_truth_arr, waypoints_predicted_arr)

    plot_cumulative(ax4, waypoints_ground_truth_arr, waypoints_predicted_arr)

    # 保存图片（如果提供路径）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

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

    data_root = DATA_PATH
    dataset = DrivingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=ref_collate_fn)

    model = load_model(config)

    # 读取 plots_path，默认保存到当前目录
    plots_path = config.get("plots_path", "./output")
    os.makedirs(plots_path, exist_ok=True)
    save_path = os.path.join(plots_path, "waypoints_visualization.png")

    visualize_model_output(model, dataloader, device, save_path)

if __name__ == "__main__":
    main()