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
        data_root 下需要包含：
          - data/objects/ (包含各个 sequence 文件夹，每个文件夹内有 imu_data.json、cametra_interface_output.csv、cametra_interface_lanes_output.csv)
          - data/waypoints/ (包含各个 sequence 文件夹，每个文件夹内有 waypoints.npy)
        """
        self.data_root = data_root
        # 全局收集所有场景的 boundary_name，确保各场景 one-hot 编码一致
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
            # 加载 IMU 数据
            imu_path = os.path.join(objects_dir, seq, 'imu_data.json')
            with open(imu_path, 'r') as f:
                imu_data = json.load(f)
            
            # 加载 Waypoints 数据 (形状为 [120, 4, 2])
            waypoints_path = os.path.join(data_root, 'waypoints', seq, 'waypoints.npy')
            waypoints = np.load(waypoints_path)
            
            # 加载 Objects 数据
            objects_path = os.path.join(objects_dir, seq, 'cametra_interface_output.csv')
            df_objects = pd.read_csv(objects_path)
            # 假设 'name' 列表示帧号
            objects_grouped = df_objects.groupby('name')
            
            # 加载 Lanes 数据
            lanes_path = os.path.join(objects_dir, seq, 'cametra_interface_lanes_output.csv')
            df_lanes = pd.read_csv(lanes_path)
            lanes_grouped = df_lanes.groupby('frame_id')
            
            # 以 imu_data 的时间戳为主（确保每个时间戳都有数据）
            timestamps = sorted(list(imu_data.keys()), key=lambda x: float(x))
            for i, ts in enumerate(timestamps):
                # IMU 数据：取 'vf'
                imu_feat = np.array([imu_data[ts]['vf']], dtype=np.float32)
                
                # Objects 数据：选取 ['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']
                if ts in objects_grouped.groups:
                    objs = objects_grouped.get_group(ts)
                    obj_features = objs[['lat_dist', 'long_dist', 'abs_vel_x', 'abs_vel_z']].values.astype(np.float32)
                else:
                    # 返回一个默认的零向量，避免空张量
                    obj_features = np.zeros((1, 4), dtype=np.float32)

                # Lanes 数据：one-hot 编码 boundary_name + 多项式系数
                if ts in lanes_grouped.groups:
                    lanes_group = lanes_grouped.get_group(ts)
                    lane_features = []
                    for row in lanes_group.itertuples(index=False):
                        one_hot = np.zeros(self.num_boundary_types, dtype=np.float32)
                        if row.boundary_name in self.boundary_to_idx:
                            one_hot[self.boundary_to_idx[row.boundary_name]] = 1
                        # 请确保 CSV 中的列名与此处保持一致，如 polynomial_0, polynomial_1, polynomial_2
                        poly = [row.polynomial_0, row.polynomial_1, row.polynomial_2]
                        lane_feat = np.concatenate([one_hot, np.array(poly, dtype=np.float32)])
                        lane_features.append(lane_feat)
                    lane_features = np.array(lane_features, dtype=np.float32)
                else:
                    lane_features = np.zeros((0, self.num_boundary_types + 3), dtype=np.float32)
                
                # 获取对应帧的 Waypoints (形状 (4,2))
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
    """对批次进行 padding 对齐"""
    objects = pad_sequence([torch.tensor(x['objects'], dtype=torch.float32) for x in batch], batch_first=True)
    lanes = pad_sequence([torch.tensor(x['lanes'], dtype=torch.float32) for x in batch], batch_first=True)
    imu = torch.stack([torch.tensor(x['imu'], dtype=torch.float32) for x in batch])
    waypoints = torch.stack([torch.tensor(x['waypoints'], dtype=torch.float32) for x in batch])
    # 生成 lane_mask：某一行全为0则视为无效
    lane_mask = (lanes.abs().sum(dim=2) != 0)
    return {
        'objects': objects,
        'lanes': lanes,
        'lane_mask': lane_mask,
        'imu': imu,
        'waypoints': waypoints
    }