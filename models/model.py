import torch
import torch.nn as nn
import torch.nn.functional as F

class WaypointPredictor(nn.Module):
    def __init__(self, obj_feature_size=4, lane_feature_size=6, imu_feature_size=1, 
                 obj_embedding_size=32, lane_embedding_size=32, imu_embedding_size=16,
                 context_size=128, step_embedding_size=16):
        super().__init__()
        
        # Object encoder
        self.object_mlp = nn.Sequential(
            nn.Linear(obj_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, obj_embedding_size)
        )
        
        # Lane encoder
        self.lane_mlp = nn.Sequential(
            nn.Linear(lane_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, lane_embedding_size)
        )
        
        # IMU encoder
        self.imu_mlp = nn.Sequential(
            nn.Linear(imu_feature_size, 16),
            nn.ReLU(),
            nn.Linear(16, imu_embedding_size)
        )
        
        # Context MLP
        self.context_mlp = nn.Sequential(
            nn.Linear(obj_embedding_size + lane_embedding_size + imu_embedding_size, context_size),
            nn.ReLU(),
            nn.Linear(context_size, context_size)
        )
        
        # Step embeddings
        self.step_embedding = nn.Embedding(4, step_embedding_size)
        
        # GRU cell
        self.gru_cell = nn.GRUCell(step_embedding_size, context_size)
        
        # Waypoint predictor
        self.waypoint_predictor = nn.Linear(context_size, 2)
        
    def forward(self, objects, lanes, imu, objects_mask, lanes_mask):
        # Process objects
        B, max_N, _ = objects.shape
        obj_flat = objects.view(B * max_N, -1)
        obj_emb_flat = self.object_mlp(obj_flat)
        obj_emb = obj_emb_flat.view(B, max_N, -1)
        obj_emb = obj_emb * objects_mask.unsqueeze(-1)
        obj_agg = torch.sum(obj_emb, dim=1) / (torch.sum(objects_mask, dim=1, keepdim=True) + 1e-6)
        
        # Process lanes
        B, max_M, _ = lanes.shape
        lane_flat = lanes.view(B * max_M, -1)
        lane_emb_flat = self.lane_mlp(lane_flat)
        lane_emb = lane_emb_flat.view(B, max_M, -1)
        lane_emb = lane_emb * lanes_mask.unsqueeze(-1)
        lane_agg = torch.sum(lane_emb, dim=1) / (torch.sum(lanes_mask, dim=1, keepdim=True) + 1e-6)
        
        # Process IMU
        imu_emb = self.imu_mlp(imu)
        
        # Context
        context = torch.cat([obj_agg, lane_agg, imu_emb], dim=1)
        context = self.context_mlp(context)
        
        # GRU steps
        h = context
        waypoints = []
        for step in range(4):
            step_idx = torch.tensor([step] * B, device=objects.device)
            step_emb = self.step_embedding(step_idx)
            h = self.gru_cell(step_emb, h)
            wp = self.waypoint_predictor(h)
            waypoints.append(wp)
        
        waypoints = torch.stack(waypoints, dim=1)
        return waypoints