import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DrivingDataset(Dataset):
    def __init__(self, data_dir, sequence_name, transform=None):
        """
        初始化数据集

        参数:
            data_dir: 数据根目录路径
            sequence_name: 场景名称，对应 data/objects/sequence_name、data/waypoints/sequence_name 下的文件
            transform: 可选的预处理方法
        """
        self.data_dir = data_dir
        self.sequence_name = sequence_name
        self.transform = transform
        
        # 加载 IMU 数据
        imu_path = os.path.join(data_dir, 'objects', sequence_name, 'imu_data.json')
        with open(imu_path, 'r') as f:
            self.imu_data = json.load(f)
        
        # 加载车道数据
        lanes_path = os.path.join(data_dir, 'objects', sequence_name, 'cametra_interface_lanes_output.csv')
        self.lanes_data = pd.read_csv(lanes_path)
        
        # 加载对象数据
        objects_path = os.path.join(data_dir, 'objects', sequence_name, 'cametra_interface_output.csv')
        self.objects_data = pd.read_csv(objects_path)
        
        # 加载航点数据
        waypoints_path = os.path.join(data_dir, 'waypoints', sequence_name, 'waypoints.npy')
        self.waypoints = np.load(waypoints_path)
        
        # 以 IMU 数据中的时间戳作为完整时间戳列表（假设 JSON 的 key 都为字符串的时间戳）
        # 根据需要转换为数字或保持字符串格式，这里假设时间戳可以直接比较
        self.timestamps = sorted(self.imu_data.keys(), key=lambda x: int(x))
    
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        # 获取对应时间戳
        timestamp = self.timestamps[idx]
        
        # 从 IMU 数据中获取当前帧的特征（如 vf 等）
        imu_features = self.imu_data[timestamp]
        
        # 获取对应时间戳的车道数据，CSV 中 frame_id 记录时间戳（注意类型转换）
        lanes = self.lanes_data[self.lanes_data['frame_id'] == int(timestamp)]
        # 如果没有车道数据，则返回空列表或做相应处理
        lanes_features = lanes.to_dict(orient='records') if not lanes.empty else []
        
        # 获取对应时间戳的对象数据，CSV 中的 name 对应时间戳
        objects = self.objects_data[self.objects_data['name'] == int(timestamp)]
        objects_features = objects.to_dict(orient='records') if not objects.empty else []
        
        # 获取航点数据
        # 假设 waypoints.npy 中的第 idx 行对应时间戳为 self.timestamps[idx]
        waypoint = self.waypoints[idx]  # shape (4, 2)
        
        sample = {
            'timestamp': timestamp,
            'imu': imu_features,
            'lanes': lanes_features,
            'objects': objects_features,
            'waypoints': waypoint
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# 下面是一个简单的测试代码
if __name__ == '__main__':
    # 假设数据根目录为 "./data" 且序列名称为 "sequence_name"
    dataset = DrivingDataset(data_dir='/home/tao/Documents/autobrains_home_assignment/objects_assignment/', sequence_name='1690794336000052_20230731090536-00-00')
    print("数据集大小:", len(dataset))
    sample = dataset[0]
    print("第一个样本:", sample)
