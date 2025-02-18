import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

from imitationLearning.utils.utils import make_abs_path

# Create absolute paths for key data directories
WAYPOINTS_PATH = make_abs_path(__file__, "../../../data/waypoints")
OBJECTS_PATH = make_abs_path(__file__, "../../../data/objects")
IMAGE_PATH = make_abs_path(__file__, "../../../data/images")
OUTPUT_PATH = make_abs_path(__file__, "../../../output")

def visualize_waypoints(scene: str, fig, gs) -> None:
    """
    Read `waypoints.npy` and plot:
    - `ax_main` (0,1) : **Overall Trajectory**
    - `small_axes` (1,0) : **10 small subplots, each showing 10 `waypoints`**
    - `ax_cumulative` (1,1) : **Cumulative Trajectory**
    
    Args:
        scene (str): Scene name
        fig (matplotlib.figure): Figure
        gs (gridspec.GridSpec): `GridSpec` structure
    """
    waypoints_file = os.path.join(WAYPOINTS_PATH, scene, "waypoints.npy")
    waypoints = np.load(waypoints_file, allow_pickle=True)
    
    print(f"Scene: {scene}, Waypoints shape:", waypoints.shape)  # Debug information

    # ======== Create `subplot` ========
    ax_overall = fig.add_subplot(gs[0, 1])  # Top right Overall Trajectory
    ax_cumulative = fig.add_subplot(gs[1, 1])  # Bottom right Cumulative

    # Subdivide (1,0) into 10 small `subplot`
    gs_small = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[1, 0])  
    small_axes = [fig.add_subplot(gs_small[i // 5, i % 5]) for i in range(10)]  # 2 rows 5 columns

    # ======== 1. Plot Overall Trajectory in `ax_overall` ========
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

    # ======== 2. Plot 10 sample Waypoints in `small_axes` ========
    for i, ax in enumerate(small_axes):
        if i >= waypoints.shape[0]:  
            break  # Avoid exceeding data range

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

    small_axes[0].legend()  # Only the first subplot shows the legend

    # ======== 3. Plot Cumulative Trajectory in `ax_cumulative` ========
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
    Main program: Visualization
      - Waypoints trajectory
      - IMU speed data
    """
    
    # Create output directory
    output_dir = OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} was not created successfully!")

    
    # Get all scenes
    scenes = os.listdir(WAYPOINTS_PATH)
    if not scenes:
        print("No scenes found in the waypoints directory!")
        return
    
    for scene in scenes:
        print(f"Visualizing scene: {scene}")

        # ======== 1. Create `GridSpec` layout ========
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])  # Increase space for the second row

        ax_imu = fig.add_subplot(gs[0, 0])  # Top left IMU Speed

        # ======== 2. Visualize `waypoints` and `imu` ========
        visualize_waypoints(scene, fig, gs)
        visualize_imu(scene, ax_imu)


        # ======== 3. Save image to `output` directory ========
        save_path = os.path.join(output_dir, f"{scene}_visualization.png")
        plt.savefig(save_path, dpi=300)  # Save with high resolution
        print(f"Saved visualization: {save_path}")

        # Adjust layout & display
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()