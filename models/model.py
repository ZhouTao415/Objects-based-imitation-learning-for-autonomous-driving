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
                 context_size=256,  # 增大上下文维度
                 step_embedding_size=32,
                 num_lstm_layers=3,  # LSTM层数
                 dropout_prob=0.3):   # Dropout概率
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
            nn.Dropout(dropout_prob),
            nn.Linear(context_size, context_size)
        )
        
        # 步长（每个 waypoint）嵌入
        self.step_embedding = nn.Embedding(4, step_embedding_size)
        
        # 多层LSTMCell配置：用 ModuleList 构造每一层
        self.num_layers = num_lstm_layers
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(
                input_size=step_embedding_size if i == 0 else context_size,
                hidden_size=context_size
            ) for i in range(num_lstm_layers)
        ])
        
        # 层间Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # 最终预测层，将最后一层隐藏状态映射为2D waypoint
        self.waypoint_predictor = nn.Sequential(
            nn.Linear(context_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, objects, lanes, imu, objects_mask, lanes_mask):
        B = objects.shape[0]
        
        # 处理 objects 部分
        if objects.shape[1] == 0:
            obj_agg = torch.zeros(B, self.object_mlp[-1].out_features, device=objects.device)
        else:
            B, max_N, _ = objects.shape
            obj_flat = objects.view(B * max_N, -1)
            obj_emb_flat = self.object_mlp(obj_flat)
            obj_emb = obj_emb_flat.view(B, max_N, -1)
            obj_emb = obj_emb * objects_mask.unsqueeze(-1)
            obj_agg = torch.sum(obj_emb, dim=1) / (torch.sum(objects_mask, dim=1, keepdim=True) + 1e-6)
        
        # 处理 lanes 部分
        if lanes.shape[1] == 0:
            lane_agg = torch.zeros(B, self.lane_mlp[-1].out_features, device=lanes.device)
        else:
            B, max_M, _ = lanes.shape
            lane_flat = lanes.view(B * max_M, -1)
            lane_emb_flat = self.lane_mlp(lane_flat)
            lane_emb = lane_emb_flat.view(B, max_M, -1)
            lane_emb = lane_emb * lanes_mask.unsqueeze(-1)
            lane_agg = torch.sum(lane_emb, dim=1) / (torch.sum(lanes_mask, dim=1, keepdim=True) + 1e-6)
        
        # 处理 IMU 部分
        imu_emb = self.imu_mlp(imu)
        
        # 生成上下文向量（增强特征融合）
        context = self.context_mlp(torch.cat([obj_agg, lane_agg, imu_emb], dim=1))
        
        # 初始化多层LSTM的状态：每层的隐藏状态都用 context 的拷贝；细胞状态全为0
        h = [context.clone() for _ in range(self.num_layers)]
        c = [torch.zeros_like(context) for _ in range(self.num_layers)]
        
        waypoints = []
        for step in range(4):
            # 构造当前步长的索引，并获取对应嵌入
            step_idx = torch.tensor([step] * B, device=objects.device)
            x = self.step_embedding(step_idx)  # (B, step_embedding_size)
            
            # 多层 LSTMCell 的前向传播
            for layer in range(self.num_layers):
                h_new, c_new = self.lstm_cells[layer](x, (h[layer], c[layer]))
                # 除最后一层外应用 dropout
                if layer < self.num_layers - 1:
                    h_new = self.dropout(h_new)
                h[layer] = h_new
                c[layer] = c_new
                x = h_new  # 当前层的输出作为下一层的输入
                
            # 最后一层的隐藏状态用于预测当前 timestep 的 waypoint
            wp = self.waypoint_predictor(h[-1])
            waypoints.append(wp)
        
        # 将4个 timestep 的输出堆叠成 (B, 4, 2)
        return torch.stack(waypoints, dim=1)