import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import your project modules
from imitationLearning.data_loader.data_lodaer import DrivingDataset
from imitationLearning.utils.utils import make_abs_path

# Create absolute paths for key data directories
WAYPOINTS_PATH = make_abs_path(__file__, "../data/waypoints")
OBJECTS_PATH = make_abs_path(__file__, "../data/objects")
IMAGE_PATH = make_abs_path(__file__, "../data/images")

def visualize_waypoints(scene: str, ax1, ax2) -> None:
    """
    Visualize the waypoint trajectory for a given scene.
    
    Args:
        scene (str): Scene folder name
        ax1 (matplotlib.axes): The subplot for the first trajectory plot
        ax2 (matplotlib.axes): The subplot for the overall trajectory plot
    """
    waypoints_file = os.path.join(WAYPOINTS_PATH, scene, "waypoints.npy")
    waypoints = np.load(waypoints_file, allow_pickle=True)
    print(np.shape(waypoints))

    n = 0
    ax1.plot(
        np.cumsum(waypoints[n, :, 0]),
        np.cumsum(waypoints[n, :, 1]),
        marker="o",
        linestyle="-",
        color="b",
    )
    ax1.set_title(f"Trajectory_{n}")
    ax1.set_ylabel("Longitudinal Distance")
    ax1.set_xlabel("Lateral Distance")
    ax1.grid(True)

    waypoints = np.reshape(waypoints, (-1, 2))
    
    ax2.plot(
        np.cumsum(waypoints[:, 0]),
        np.cumsum(waypoints[:, 1]),
        linestyle="-",
        color="b",
    )
    ax2.set_title("Overall Trajectory")
    ax2.set_ylabel("Longitudinal Distance")
    ax2.set_xlabel("Lateral Distance")
    ax2.grid(True)



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
    Main entry point for data visualization.
    
    This script visualizes:
      1. Waypoint trajectories,
      2. IMU speed over time,
    for a selected scene.
    """
    # Determine the list of available scenes from the waypoints directory
    scenes = os.listdir(WAYPOINTS_PATH)
    if not scenes:
        print("No scenes found in the waypoints directory!")
        return
    
    # Iterate through each scene to visualize
    for scene in scenes:
        print(f"Visualizing scene: {scene}")

        # Create a 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Call visualization functions and pass subplots
        visualize_waypoints(scene, axes[0, 0], axes[0, 1])  # First row, two subplots
        visualize_imu(scene, axes[1, 0])  # Second row, left subplot

        # Adjust layout for better display
        plt.tight_layout()
        plt.show()  # Call show() only once to display all plots together

if __name__ == "__main__":
    main()
