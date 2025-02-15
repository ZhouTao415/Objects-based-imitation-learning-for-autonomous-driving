import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DrivingDataset(Dataset):
    def __init__(self,data_root):
        """ 
        Adapt the data interface:
            - Maintain the original features processing logic
            - Align the data directory structure 
        """
        self.data_root = data_root
        # Load the sample data
        self.samples = self._load_samples(data_root)
    
    def _load_samples(self, data_root):
        samples = []
        
        # traversal data directory
        objects_dir = os.path.join(data_root, 'objects')
        sequences = [d for d in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, d))]
        
        for seq in sequences:
            # Load the IMU data
            imu_path = os.path.join(objects_dir, seq, 'imu_data.json')
            with open(imu_path, 'r') as f:
                imu_data = json.load(f)
            
            # Load the waypoint data
            waypoints_path = os.path.join(data_root, 'waypoints', seq, 'waypoints.npy')
            waypoints = np.load(waypoints_path)
            
            # Load the objects data df:data frame
            objects_path = os.path.join(objects_dir, seq, 'cametra_interface_output.csv')
            df_objects = pd.read_csv(objects_path)
            objects_grouped = df_objects.groupby('name')
            
            # Load the lanes data
            lanes_path = os.path.join(objects_dir, seq, 'cametra_interface_lanes_output.csv')
            df_lanes = pd.read_csv(lanes_path)
            lanes_grouped = df_lanes.groupby('frame_id')
            
            # traversal all the timestamps (based on the IMU data)
            timestamps = sorted(imu_data.key(), key = lambda x : float(x))
            
            for i, ts in enumerate(timestamps):
                # IMU data: features 'vf'
                imu_feat = np.array([imu_data[ts]['vf']], dtype=np.float32)
                
                # Objects data: features ['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']
                if ts in objects_grouped.groups:
                    objs = objects_grouped.get_group(ts)
                    obj_features = objs[['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']].values.astype(np.float32)
                else:
                    obj_features = np.zeros((0, 4), dtype=np.float32)
                
                # Lane data: one-hot encode boundary_name and concatenate polynomial coefficients
                boundary_names = df_lanes['boundary_name'].unique()
                boundary_to_idx = {name: i for i, name in enumerate(sorted(boundary_names))}
                num_boundary_types = len(boundary_to_idx)
                if ts in lanes_grouped.groups:
                    lanes_group = lanes_grouped.get_group(ts)
                    lane_features = []
                    for row in lanes_group.itertuples(index=False):
                        one_hot = np.zeros(num_boundary_types, dtype=np.float32)
                        if row['boundary_name'] in boundary_to_idx:
                            one_hot[boundary_to_idx[row['boundary_name']]] = 1
                        poly = [row['polynomial_0'], row['polynomial_1'], row['polynomial_2']]
                        lane_feat = np.concatenate([one_hot, np.array(poly, dtype=np.float32)])
                        lane_features.append(lane_feat)
                    lane_features = np.array(lane_features, dtype=np.float32)
                else:
                    # One-hot encoded categories + 3 polynomial coefficients
                    lane_features = np.zeros((0, num_boundary_types + 3), dtype=np.float32)
                    
                # Waypoints data: features [120, 4, 2]
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
        """Align padding strategy"""
        objects = pad_sequence([torch.tensor(x['objects']) for x in batch], batch_first=True)
        lanes = pad_sequence([torch.tensor(x['lanes']) for x in batch], batch_first=True)
        imu = torch.stack([torch.tensor(x['imu']) for x in batch])
        waypoints = torch.stack([torch.tensor(x['waypoints']) for x in batch])
        return {
            'objects': objects,
            'lanes': lanes,
            'imu': imu,
            'waypoints': waypoints
        }