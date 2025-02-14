import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class DrivingDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.sequences = os.listdir(os.path.join(data_root, 'objects'))
        self.boundary_names = self._get_boundary_names()
        self.num_boundary_types = len(self.boundary_names)
        self.boundary_to_idx = {name: i for i, name in enumerate(self.boundary_names)}
        self.samples = self._load_samples()
        
    def _get_boundary_names(self):
        boundary_names = set()
        for seq in self.sequences:
            lanes_path = os.path.join(self.data_root, 'objects', seq, 'cametra_interface_lanes_output.csv')
            df = pd.read_csv(lanes_path)
            boundary_names.update(df['boundary_name'].unique())
        return sorted(list(boundary_names))
    
    def _load_samples(self):
        samples = []
        for seq in self.sequences:
            # Load IMU data
            imu_path = os.path.join(self.data_root, 'objects', seq, 'imu_data.json')
            with open(imu_path, 'r') as f:
                imu_data = json.load(f)
            # Load waypoints
            waypoints_path = os.path.join(self.data_root, 'waypoints', seq, 'waypoints.npy')
            waypoints = np.load(waypoints_path)
            # Load objects
            objects_path = os.path.join(self.data_root, 'objects', seq, 'cametra_interface_output.csv')
            df_objects = pd.read_csv(objects_path)
            objects_grouped = df_objects.groupby('name')
            # Load lanes
            lanes_path = os.path.join(self.data_root, 'objects', seq, 'cametra_interface_lanes_output.csv')
            df_lanes = pd.read_csv(lanes_path)
            lanes_grouped = df_lanes.groupby('frame_id')
            # Process each timestamp
            timestamps = sorted(imu_data.keys(), key=lambda x: float(x))
            for i, ts in enumerate(timestamps):
                # IMU
                imu = np.array([imu_data[ts]['vf']], dtype=np.float32)
                # Objects
                if ts in objects_grouped.groups:
                    objs = objects_grouped.get_group(ts)
                    obj_features = objs[['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']].values.astype(np.float32)
                else:
                    obj_features = np.zeros((0, 4), dtype=np.float32)
                # Lanes
                if ts in lanes_grouped.groups:
                    lanes = lanes_grouped.get_group(ts)
                    lane_features = []
                    for _, row in lanes.iterrows():
                        one_hot = np.zeros(self.num_boundary_types)
                        if row['boundary_name'] in self.boundary_to_idx:
                            one_hot[self.boundary_to_idx[row['boundary_name']]] = 1
                        poly = [row['polynomial_0'], row['polynomial_1'], row['polynomial_2']]
                        lane_features.append(np.concatenate([one_hot, poly]))
                    lane_features = np.array(lane_features, dtype=np.float32)
                else:
                    lane_features = np.zeros((0, self.num_boundary_types + 3), dtype=np.float32)
                # Waypoints
                wp = waypoints[i]
                samples.append({
                    'objects': obj_features,
                    'lanes': lane_features,
                    'imu': imu,
                    'waypoints': wp
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    objects = [torch.tensor(sample['objects']) for sample in batch]
    lanes = [torch.tensor(sample['lanes']) for sample in batch]
    imu = torch.tensor([sample['imu'] for sample in batch], dtype=torch.float32)
    waypoints = torch.tensor([sample['waypoints'] for sample in batch], dtype=torch.float32)
    
    # Pad objects
    max_objects = max(o.shape[0] for o in objects)
    padded_objects = []
    objects_mask = []
    for obj in objects:
        padding = (0, 0, 0, max_objects - obj.shape[0])
        padded = F.pad(obj, padding)
        padded_objects.append(padded)
        mask = [1] * obj.shape[0] + [0] * (max_objects - obj.shape[0])
        objects_mask.append(mask)
    padded_objects = torch.stack(padded_objects)
    objects_mask = torch.tensor(objects_mask, dtype=torch.float32)
    
    # Pad lanes
    max_lanes = max(l.shape[0] for l in lanes)
    padded_lanes = []
    lanes_mask = []
    for ln in lanes:
        padding = (0, 0, 0, max_lanes - ln.shape[0])
        padded = F.pad(ln, padding)
        padded_lanes.append(padded)
        mask = [1] * ln.shape[0] + [0] * (max_lanes - ln.shape[0])
        lanes_mask.append(mask)
    padded_lanes = torch.stack(padded_lanes)
    lanes_mask = torch.tensor(lanes_mask, dtype=torch.float32)
    
    return {
        'objects': padded_objects,
        'lanes': padded_lanes,
        'imu': imu,
        'waypoints': waypoints,
        'objects_masFk': objects_mask,
        'lanes_mask': lanes_mask
    }