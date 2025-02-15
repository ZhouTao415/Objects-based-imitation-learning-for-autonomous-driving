import numpy as np
import matplotlib.pyplot as plt

# 加载数据
pred = np.load("/home/tao/Documents/autobrains_home_assignment/predictions/pred_waypoints_20250214-152324.npy")
# true = np.load("/home/tao/Documents/autobrains_home_assignment/data/waypoints/1690794336000052_20230731090536-00-00/waypoints.npy")
true = np.load("/home/tao/Documents/autobrains_home_assignment/data/waypoints/1692632429000129_20230821154029-00-00/waypoints.npy")

import time

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 6))

for idx in range(pred.shape[0]):  # Iterate over all samples
    ax.clear()  # Clear previous plot

    # Plot true waypoints
    for i in range(true.shape[1]):
        ax.plot(true[idx, i, 1], true[idx, i, 0], 'go', label="Ground Truth" if i == 0 else "", markersize=5)

    # Plot predicted waypoints
    for i in range(pred.shape[1]):
        ax.plot(pred[idx, i, 1], pred[idx, i, 0], 'rx', label="Prediction" if i == 0 else "", markersize=5)

    ax.set_xlabel("Longitudinal Distance (m)")
    ax.set_ylabel("Lateral Distance (m)")
    ax.set_title(f"Waypoints Comparison - Sample {idx+1}/{pred.shape[0]}")
    ax.legend()
    ax.grid(True)

    plt.pause(0.5)  # Pause for 0.5 seconds to simulate animation

plt.show()
