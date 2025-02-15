import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class DrivingDataset(Dataset):
    def __init__(self, data_root):
        """
        data_root: 数据根目录，假设目录结构为：
          data/
            ├── images/sequence_name/
            ├── waypoints/sequence_name/waypoints.npy
            └── objects/sequence_name/ (包含 cametra_interface_output.csv, cametra_interface_lanes_output.csv, imu_data.json)
        """
        self.data_root = data_root
        # 获取所有场景名称（例如目录 names under data/objects）
        self.sequences = os.listdir(os.path.join(data_root, 'objects'))
        
        # 预先获取所有 lane 的类别，用于 one-hot 编码
        self.boundary_names = self._get_boundary_names()
        self.num_boundary_types = len(self.boundary_names)
        self.boundary_to_idx = {name: i for i, name in enumerate(self.boundary_names)}
        
        # 遍历所有场景，加载每个时间戳对应的数据
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
            # 读取 IMU 数据
            imu_path = os.path.join(self.data_root, 'objects', seq, 'imu_data.json')
            with open(imu_path, 'r') as f:
                imu_data = json.load(f)
            # 读取 waypoints 数据
            waypoints_path = os.path.join(self.data_root, 'waypoints', seq, 'waypoints.npy')
            waypoints_arr = np.load(waypoints_path)  # (120,4,2)
            # 读取 objects 数据
            objects_path = os.path.join(self.data_root, 'objects', seq, 'cametra_interface_output.csv')
            df_objects = pd.read_csv(objects_path)
            objects_grouped = df_objects.groupby('name')
            # 读取 lanes 数据
            lanes_path = os.path.join(self.data_root, 'objects', seq, 'cametra_interface_lanes_output.csv')
            df_lanes = pd.read_csv(lanes_path)
            lanes_grouped = df_lanes.groupby('frame_id')
            
            # 按照 IMU 的时间戳排序（假设 key 可转为浮点数）
            timestamps = sorted(imu_data.keys(), key=lambda x: float(x))
            for i, ts in enumerate(timestamps):
                # IMU 特征
                imu_feature = np.array([imu_data[ts]['vf']], dtype=np.float32)
                
                # objects 特征
                if ts in objects_grouped.groups:
                    group = objects_grouped.get_group(ts)
                    # 选择需要的列：lat_dist, long_dist, abs_vel_x, abs_vel_z
                    obj_features = group[['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']].values.astype(np.float32)
                else:
                    obj_features = np.zeros((0, 4), dtype=np.float32)
                    
                # lanes 特征
                if ts in lanes_grouped.groups:
                    group = lanes_grouped.get_group(ts)
                    lane_features = []
                    for _, row in group.iterrows():
                        # 将 boundary_name one-hot 编码
                        one_hot = np.zeros(self.num_boundary_types, dtype=np.float32)
                        if row['boundary_name'] in self.boundary_to_idx:
                            one_hot[self.boundary_to_idx[row['boundary_name']]] = 1.0
                        # 获取多项式系数
                        poly = np.array([row['polynomial_0'], row['polynomial_1'], row['polynomial_2']], dtype=np.float32)
                        lane_feature = np.concatenate([one_hot, poly])
                        lane_features.append(lane_feature)
                    lane_features = np.array(lane_features, dtype=np.float32)
                else:
                    lane_features = np.zeros((0, self.num_boundary_types + 3), dtype=np.float32)
                    
                # 对应的 waypoints
                wp = waypoints_arr[i]  # (4,2)
                
                samples.append({
                    'objects': obj_features,   # shape: (N,4)
                    'lanes': lane_features,      # shape: (M,6)
                    'imu': imu_feature,          # shape: (1,)
                    'waypoints': wp              # shape: (4,2)
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """
    将 batch 中每个样本里的 objects 与 lanes 用 0 padding 到同一尺寸，并生成 mask。
    返回一个字典，包含 padded tensors 及 mask。
    """
    # 利用 from_numpy 转换
    objects_list = [torch.from_numpy(sample['objects']) for sample in batch]
    lanes_list = [torch.from_numpy(sample['lanes']) for sample in batch]
    imu = torch.from_numpy(np.array([sample['imu'] for sample in batch])).float()
    waypoints = torch.from_numpy(np.array([sample['waypoints'] for sample in batch])).float()
    
    # 计算所有样本中 objects 和 lanes 的最大行数
    max_objects = max(o.shape[0] for o in objects_list)
    max_lanes = max(l.shape[0] for l in lanes_list)
    
    padded_objects = []
    objects_mask = []
    for o in objects_list:
        num_obj = o.shape[0]
        # 如果为空，则直接创建一个空 tensor
        if num_obj == 0:
            padded = torch.zeros(max_objects, o.shape[1] if o.ndim > 1 else 1)
            mask = [0] * max_objects
        else:
            pad = (0, 0, 0, max_objects - num_obj)
            padded = F.pad(o, pad, "constant", 0)
            mask = [1] * num_obj + [0] * (max_objects - num_obj)
        padded_objects.append(padded)
        objects_mask.append(mask)
    padded_objects = torch.stack(padded_objects)
    objects_mask = torch.tensor(objects_mask, dtype=torch.float32)
    
    padded_lanes = []
    lanes_mask = []
    for ln in lanes_list:
        num_ln = ln.shape[0]
        if num_ln == 0:
            padded = torch.zeros(max_lanes, ln.shape[1] if ln.ndim > 1 else 1)
            mask = [0] * max_lanes
        else:
            pad = (0, 0, 0, max_lanes - num_ln)
            padded = F.pad(ln, pad, "constant", 0)
            mask = [1] * num_ln + [0] * (max_lanes - num_ln)
        padded_lanes.append(padded)
        lanes_mask.append(mask)
    padded_lanes = torch.stack(padded_lanes)
    lanes_mask = torch.tensor(lanes_mask, dtype=torch.float32)
    
    return {
        'objects': padded_objects,      # (B, max_objects, 4)
        'lanes': padded_lanes,            # (B, max_lanes, 6)
        'imu': imu,                     # (B, 1)
        'waypoints': waypoints,         # (B, 4, 2)
        'objects_mask': objects_mask,   # (B, max_objects)
        'lanes_mask': lanes_mask        # (B, max_lanes)
    }