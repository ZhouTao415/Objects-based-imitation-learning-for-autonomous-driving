import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DrivingDataset(Dataset):
    def __init__(self, data_root):
        """
        The data_root should contain:
          - data/objects/ (contains sequence folders, each with imu_data.json, cametra_interface_output.csv, cametra_interface_lanes_output.csv)
          - data/waypoints/ (contains sequence folders, each with waypoints.npy)
        """
        self.data_root = data_root
        # Collect all boundary names globally to ensure consistent one-hot encoding across scenes
        self.boundary_names = self._get_all_boundary_names(data_root)
        self.num_boundary_types = len(self.boundary_names)
        self.boundary_to_idx = {name: i for i, name in enumerate(self.boundary_names)}
        self.samples = self._load_samples(data_root)
    
    def _get_all_boundary_names(self, data_root):
        boundary_names = set()
        objects_dir = os.path.join(data_root, 'objects')
        sequences = [d for d in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, d))]
        for seq in sequences:
            lanes_path = os.path.join(objects_dir, seq, 'cametra_interface_lanes_output.csv')
            df = pd.read_csv(lanes_path)
            boundary_names.update(df['boundary_name'].unique())
        return sorted(list(boundary_names))
    
    def _load_samples(self, data_root):
        samples = []
        objects_dir = os.path.join(data_root, 'objects')
        sequences = [d for d in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, d))]
        
        for seq in sequences:
            # Load IMU data
            imu_path = os.path.join(objects_dir, seq, 'imu_data.json')
            # print(f"Loading IMU data from: {imu_path}")
            with open(imu_path, 'r') as f:
                imu_data = json.load(f)
            
            # Load Waypoints data (shape [120, 4, 2])
            waypoints_path = os.path.join(data_root, 'waypoints', seq, 'waypoints.npy')
            # print(f"Loading waypoints from: {waypoints_path}")
            waypoints = np.load(waypoints_path)
            
            # Load Objects data
            objects_path = os.path.join(objects_dir, seq, 'cametra_interface_output.csv')
            # print(f"Loading objects from: {objects_path}")
            df_objects = pd.read_csv(objects_path)
            # Assume 'name' column represents frame number
            objects_grouped = df_objects.groupby('name')
            
            # Load Lanes data
            lanes_path = os.path.join(objects_dir, seq, 'cametra_interface_lanes_output.csv')
            # print(f"Loading lanes from: {lanes_path}")
            df_lanes = pd.read_csv(lanes_path)
            lanes_grouped = df_lanes.groupby('frame_id')
            
            # Use imu_data timestamps as the primary key (ensure each timestamp has data)
            timestamps = sorted(list(imu_data.keys()), key=lambda x: float(x))
            for i, ts in enumerate(timestamps):
                # IMU data: take 'vf'
                imu_feat = np.array([imu_data[ts]['vf']], dtype=np.float32)
                
                # Objects data: select ['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']
                if ts in objects_grouped.groups:
                    objs = objects_grouped.get_group(ts)
                    obj_features = objs[['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']].values.astype(np.float32)
                else:
                    # Return a default zero vector to avoid empty tensor
                    obj_features = np.zeros((1, 4), dtype=np.float32)

                # Lanes data: one-hot encode boundary_name + polynomial coefficients
                if ts in lanes_grouped.groups:
                    lanes_group = lanes_grouped.get_group(ts)
                    lane_features = []
                    for row in lanes_group.itertuples(index=False):
                        one_hot = np.zeros(self.num_boundary_types, dtype=np.float32)
                        if row.boundary_name in self.boundary_to_idx:
                            one_hot[self.boundary_to_idx[row.boundary_name]] = 1
                        # Ensure the column names in the CSV match here, such as polynomial_0, polynomial_1, polynomial_2
                        poly = [row.polynomial_0, row.polynomial_1, row.polynomial_2]
                        lane_feat = np.concatenate([one_hot, np.array(poly, dtype=np.float32)])
                        lane_features.append(lane_feat)
                    lane_features = np.array(lane_features, dtype=np.float32)
                else:
                    lane_features = np.zeros((0, self.num_boundary_types + 3), dtype=np.float32)
                
                # Get the corresponding frame's Waypoints (shape (4,2))
                wp = waypoints[i]
                
                samples.append({
                    'objects': obj_features,
                    'lanes': lane_features,
                    'imu': imu_feat,
                    'waypoints': wp
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            'objects': sample['objects'],
            'lanes': sample['lanes'],
            'imu': sample['imu'],
            'waypoints': sample['waypoints']
        }
        
def ref_collate_fn(batch):
    """Pad the batch for alignment"""
    objects = pad_sequence([torch.tensor(x['objects'], dtype=torch.float32) for x in batch], batch_first=True)
    lanes = pad_sequence([torch.tensor(x['lanes'], dtype=torch.float32) for x in batch], batch_first=True)
    imu = torch.stack([torch.tensor(x['imu'], dtype=torch.float32) for x in batch])
    waypoints = torch.stack([torch.tensor(x['waypoints'], dtype=torch.float32) for x in batch])
    # Generate lane_mask: a row with all zeros is considered invalid
    lane_mask = (lanes.abs().sum(dim=2) != 0)
    return {
        'objects': objects,
        'lanes': lanes,
        'lane_mask': lane_mask,
        'imu': imu,
        'waypoints': waypoints
    }