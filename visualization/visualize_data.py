import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec

# Import your project modules
from imitationLearning.data_loader.data_loader import DrivingDataset
from imitationLearning.utils.utils import make_abs_path

# Create absolute paths for key data directories
WAYPOINTS_PATH = make_abs_path(__file__, "../data/waypoints")
OBJECTS_PATH = make_abs_path(__file__, "../data/objects")
IMAGE_PATH = make_abs_path(__file__, "../data/images")

def visualize_waypoints(scene: str, fig, gs) -> None:
    """
    读取 `waypoints.npy` 并绘制：
    - `ax_main` (0,1) : **Overall Trajectory**
    - `small_axes` (1,0) : **10 个小 subplot，分别显示 10 个 `waypoints`**
    - `ax_cumulative` (1,1) : **Cumulative Trajectory**
    
    Args:
        scene (str): 场景名称
        fig (matplotlib.figure): 画布
        gs (gridspec.GridSpec): `GridSpec` 结构
    """
    waypoints_file = os.path.join(WAYPOINTS_PATH, scene, "waypoints.npy")
    waypoints = np.load(waypoints_file, allow_pickle=True)
    
    print(f"Scene: {scene}, Waypoints shape:", waypoints.shape)  # 调试信息

    # ======== 创建 `subplot` ========
    ax_overall = fig.add_subplot(gs[0, 1])  # 右上角 Overall Trajectory
    ax_cumulative = fig.add_subplot(gs[1, 1])  # 右下角 Cumulative

    # 细分 (1,0) 为 10 个小 `subplot`
    gs_small = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[1, 0])  
    small_axes = [fig.add_subplot(gs_small[i // 5, i % 5]) for i in range(10)]  # 2行5列

    # ======== 1. 在 `ax_overall` 绘制 Overall Trajectory ========
    for n in range(min(20, waypoints.shape[0])):
        ax_overall.plot(
            waypoints[n, :, 0],
            waypoints[n, :, 1],
            linestyle="-",
            color="b",
            alpha=0.7,
        )
    ax_overall.set_title("Overall Trajectory")
    ax_overall.set_ylabel("Longitudinal Distance")
    ax_overall.set_xlabel("Lateral Distance")
    ax_overall.grid(True)

    # ======== 2. 在 `small_axes` 绘制 10 个样本 Waypoints ========
    for i, ax in enumerate(small_axes):
        if i >= waypoints.shape[0]:  
            break  # 避免超出数据范围

        ax.plot(
            waypoints[i, :, 0],
            waypoints[i, :, 1],
            marker="o",
            linestyle="-",
            color="b",
            label="Ground Truth" if i == 0 else "",
        )
        ax.set_xlabel("Lat Dist")
        ax.set_ylabel("Long Dist")
        ax.grid(True)

    small_axes[0].legend()  # 仅第一个子图显示 legend

    # ======== 3. 在 `ax_cumulative` 绘制 Cumulative Trajectory ========
    waypoints_flat = np.reshape(waypoints, (-1, 2))  # (N*T, 2)
    
    ax_cumulative.plot(
        np.cumsum(waypoints_flat[:, 0]),
        np.cumsum(waypoints_flat[:, 1]),
        linestyle="-",
        color="b",
    )
    ax_cumulative.set_title("Cumulative Trajectory")
    ax_cumulative.set_ylabel("Longitudinal Distance")
    ax_cumulative.set_xlabel("Lateral Distance")
    ax_cumulative.grid(True)



def visualize_imu(scene: str, ax) -> None:
    """
    Visualize the IMU speed over time for a given scene.
    
    Args:
        scene (str): Scene folder name
        ax (matplotlib.axes): The subplot to display IMU speed plot
    """
    imu_file = os.path.join(OBJECTS_PATH, scene, "imu_data.json")
    with open(imu_file, "r") as f:
        imu_data = json.load(f)
    
    timestamps = sorted(imu_data.keys(), key=lambda x: float(x))
    speed = [imu_data[ts]["vf"] for ts in timestamps]
    time_arr = np.arange(len(speed))
    
    ax.plot(time_arr, speed, marker="o", linestyle="-", color="g")
    ax.set_title(f"IMU Speed vs. Time for Scene: {scene}")
    ax.set_xlabel("Time (Frame Index)")
    ax.set_ylabel("Speed (vf)")
    ax.grid(True)



def main():
    """
    主程序：可视化
      - Waypoints 轨迹
      - IMU 速度数据
    """
    
    # 创建 output 目录
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  # 确保 output 目录存在
    
    # 获取所有场景
    scenes = os.listdir(WAYPOINTS_PATH)
    if not scenes:
        print("No scenes found in the waypoints directory!")
        return
    
    for scene in scenes:
        print(f"Visualizing scene: {scene}")

        # ======== 1. 创建 `GridSpec` 布局 ========
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])  # 增加第二行空间

        ax_imu = fig.add_subplot(gs[0, 0])  # 左上角 IMU Speed

        # ======== 2. 可视化 `waypoints` 和 `imu` ========
        visualize_waypoints(scene, fig, gs)
        visualize_imu(scene, ax_imu)


        # ======== 3. 保存图片到 `output` 目录 ========
        save_path = os.path.join(output_dir, f"{scene}_visualization.png")
        # plt.savefig(save_path, dpi=300)  # 高分辨率保存
        print(f"Saved visualization: {save_path}")

        # 调整布局 & 显示
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()