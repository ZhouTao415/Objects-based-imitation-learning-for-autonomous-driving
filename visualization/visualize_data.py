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

def visualize_waypoints(scene: str) -> None:
    """
    Visualize the waypoint trajectory for a given scene.
    
    This function loads the 'waypoints.npy' file from the scene folder,
    selects the first timestamp's set of 4 waypoints, computes the cumulative
    sum to form a trajectory, and then plots it.
    
    Args:
        scene (str): Scene folder name (e.g., "scene_001")
    """
    waypoints_file = os.path.join(WAYPOINTS_PATH, scene, "waypoints.npy")
    waypoints = np.load(waypoints_file, allow_pickle=True)
    print(np.shape(waypoints))
    
    n = 0
    plt.plot(
        np.cumsum(waypoints[n, :, 0]),
        np.cumsum(waypoints[n, :, 1]),
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.title(f"Trajectory_{n}")
    plt.ylabel("Longitudinal Distance")
    plt.xlabel("Lateral Distance")
    plt.grid(True)
    plt.show()

    waypoints = np.reshape(waypoints, (-1, 2))
    timestamp = np.arange(len(waypoints))

    plt.plot(
        np.cumsum(waypoints[:, 0]),
        np.cumsum(waypoints[:, 1]),
        linestyle="-",
        color="b",
    )
    plt.title("Overall Trajectory")
    plt.ylabel("Longitudinal Distance")
    plt.xlabel("Lateral Distance")
    plt.grid(True)
    plt.show()


def visualize_imu(scene: str) -> None:
    """
    Visualize the IMU data (e.g., forward velocity 'vf') for a given scene.
    
    This function loads the 'imu_data.json' file from the scene folder under
    the OBJECTS directory, sorts the timestamps, extracts the 'vf' (speed) values,
    and plots them against time.
    
    Args:
        scene (str): Scene folder name (e.g., "scene_001")
    """
    imu_file = os.path.join(OBJECTS_PATH, scene, "imu_data.json")
    with open(imu_file, "r") as f:
        imu_data = json.load(f)
    
    timestamps = sorted(imu_data.keys(), key=lambda x: float(x))
    speed = [imu_data[ts]["vf"] for ts in timestamps]
    time_arr = np.arange(len(speed))
    
    plt.figure()
    plt.plot(time_arr, speed, marker="o", linestyle="-", color="g")
    plt.title(f"IMU Speed vs. Time for Scene: {scene}")
    plt.xlabel("Time (Frame Index)")
    plt.ylabel("Speed (vf)")
    plt.grid(True)
    plt.show()



def main():
    """
    Main entry point for data visualization.
    
    This script visualizes:
      1. Waypoint trajectories,
      2. IMU speed over time,
      3. An example RGB image (if available)
    for a selected scene.
    """
    # Determine the list of available scenes from the waypoints directory
    scenes = os.listdir(WAYPOINTS_PATH)
    if not scenes:
        print("No scenes found in the waypoints directory!")
        return
    
    # Select a scene to visualize (e.g., the first scene)
    scene = scenes[0]
    print(f"Visualizing scene: {scene}")
    
    visualize_waypoints(scene)
    visualize_imu(scene)

if __name__ == "__main__":
    main()
