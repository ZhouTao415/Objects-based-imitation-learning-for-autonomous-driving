import torch
import torch.nn as nn
import torch.nn.functional as F

class WaypointPredictor(nn.Module):
    def __init__(self, 
                 obj_feature_size=4, 
                 lane_feature_size=6, 
                 imu_feature_size=1,
                 obj_embedding_size=32, 
                 lane_embedding_size=32, 
                 imu_embedding_size=16,
                 context_size=128, 
                 step_embedding_size=16):
        super(WaypointPredictor, self).__init__()
        
        # 对 objects 特征进行编码
        self.object_mlp = nn.Sequential(
            nn.Linear(obj_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, obj_embedding_size)
        )
        
        # 对 lanes 特征进行编码
        self.lane_mlp = nn.Sequential(
            nn.Linear(lane_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, lane_embedding_size)
        )
        
        # 对 IMU 特征进行编码
        self.imu_mlp = nn.Sequential(
            nn.Linear(imu_feature_size, 16),
            nn.ReLU(),
            nn.Linear(16, imu_embedding_size)
        )
        
        # 融合三个编码后的特征，生成上下文向量
        self.context_mlp = nn.Sequential(
            nn.Linear(obj_embedding_size + lane_embedding_size + imu_embedding_size, context_size),
            nn.ReLU(),
            nn.Linear(context_size, context_size)
        )
        
        # 步长（每个 waypoint）嵌入
        self.step_embedding = nn.Embedding(4, step_embedding_size)
        
        # GRU cell 用于生成 waypoints
        self.gru_cell = nn.GRUCell(step_embedding_size, context_size)
        
        # 隐状态到 2D waypoint 的映射
        self.waypoint_predictor = nn.Linear(context_size, 2)
        
    def forward(self, objects, lanes, imu, objects_mask, lanes_mask):
        # objects: (B, max_N, 4)
        # lanes: (B, max_M, 6)
        # imu: (B, 1)
        
        B = objects.shape[0]
        
        # 对 objects 部分
        if objects.shape[1] == 0:
            # 如果当前 batch 中所有样本都没有 objects，
            # 则直接构造一个全零的对象聚合向量，尺寸为 (B, obj_embedding_size)
            obj_agg = torch.zeros(B, self.object_mlp[-1].out_features, device=objects.device)
        else:
            B, max_N, _ = objects.shape
            obj_flat = objects.view(B * max_N, -1)
            obj_emb_flat = self.object_mlp(obj_flat)  # (B*max_N, 32)
            obj_emb = obj_emb_flat.view(B, max_N, -1)   # (B, max_N, 32)
            obj_emb = obj_emb * objects_mask.unsqueeze(-1)
            obj_agg = torch.sum(obj_emb, dim=1) / (torch.sum(objects_mask, dim=1, keepdim=True) + 1e-6)
        
        # 对 lanes 部分
        if lanes.shape[1] == 0:
            lane_agg = torch.zeros(B, self.lane_mlp[-1].out_features, device=lanes.device)
        else:
            B, max_M, _ = lanes.shape
            lane_flat = lanes.view(B * max_M, -1)
            lane_emb_flat = self.lane_mlp(lane_flat)  # (B*max_M, 32)
            lane_emb = lane_emb_flat.view(B, max_M, -1)
            lane_emb = lane_emb * lanes_mask.unsqueeze(-1)
            lane_agg = torch.sum(lane_emb, dim=1) / (torch.sum(lanes_mask, dim=1, keepdim=True) + 1e-6)
        
        # IMU 编码
        imu_emb = self.imu_mlp(imu)  # (B, 16)
        
        # 融合得到上下文向量
        context = torch.cat([obj_agg, lane_agg, imu_emb], dim=1)
        context = self.context_mlp(context)  # (B, 128)
        
        # 用 GRUCell 依次生成 4 个 waypoint
        h = context  # 初始隐状态
        waypoints = []
        for step in range(4):
            step_idx = torch.tensor([step] * B, device=objects.device)
            step_emb = self.step_embedding(step_idx)  # (B, step_embedding_size)
            h = self.gru_cell(step_emb, h)  # (B, 128)
            wp = self.waypoint_predictor(h)  # (B, 2)
            waypoints.append(wp)
        
        # 将 4 个 waypoint 堆叠起来，形状为 (B, 4, 2)
        waypoints = torch.stack(waypoints, dim=1)  # (B, 4, 2)
        return waypoints