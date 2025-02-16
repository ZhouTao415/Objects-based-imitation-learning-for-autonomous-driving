import torch
import torch.nn as nn

# Transformer Encoder，用于处理变长的 Objects 数据
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        # 采用 batch_first=True，直接传入 (B, L, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        if x.size(1) == 0:
            return torch.zeros(x.size(0), self.embedding.out_features, device=x.device)
        # x: (B, L, input_dim)
        x = self.embedding(x)            # (B, L, embed_dim)
        x = self.transformer(x)          # (B, L, embed_dim)
        return x.mean(dim=1)             # 池化成 (B, embed_dim)

# MLP Encoder，用于处理固定输入的 IMU 数据
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)

# Lane Encoder：使用 Transformer 对车道数据进行聚合
class LaneEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=2, ff_dim=64, num_layers=1):
        super(LaneEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        # x: (B, L_lane, lane_dim)
        # mask: (B, L_lane) 布尔型，True 表示有效数据
        x = self.embedding(x)         # (B, L_lane, embed_dim)
        # Transformer 的 src_key_padding_mask 要求 True 表示需要忽略的（padding）项，
        # 因此这里传入 ~mask（假设 mask True 表示有效数据）
        key_padding_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, L_lane, embed_dim)
        return x.mean(dim=1)          # (B, embed_dim)

# Waypoints Decoder：使用 RNN 逐步生成未来 4 个 2D 轨迹点
class WaypointsDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WaypointsDecoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (B, seq_len, input_dim) ，此处 seq_len = 4
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out)  # (B, seq_len, output_dim)

# 综合模型：Transformer 处理 Objects，LaneEncoder 处理 Lanes，MLP处理 IMU，
# 随后特征融合并由 RNN 生成 Waypoints
class TransformerRNNModel(nn.Module):
    def __init__(self, obj_dim, lane_dim, imu_dim, 
                 embed_dim, num_heads, ff_dim, num_layers, 
                 hidden_dim, output_dim):
        super(TransformerRNNModel, self).__init__()
        self.obj_encoder = TransformerEncoder(obj_dim, embed_dim, num_heads, ff_dim, num_layers)
        self.lane_encoder = LaneEncoder(lane_dim, embed_dim)
        self.imu_encoder = MLPEncoder(imu_dim, embed_dim)
        # 简单地将三部分特征直接拼接
        self.decoder = WaypointsDecoder(embed_dim * 3, hidden_dim, output_dim)

    def forward(self, obj_data, lanes, lane_mask, imu):
        # obj_data: (B, L_obj, obj_dim)
        # lanes: (B, L_lane, lane_dim)
        # lane_mask: (B, L_lane) 布尔型
        # imu: (B, imu_dim)
        obj_features = self.obj_encoder(obj_data)      # (B, embed_dim)
        # 对 lanes 使用 LaneEncoder；若没有车道数据，则用0填充
        if lanes.size(1) > 0:
            lane_features = self.lane_encoder(lanes, lane_mask)  # (B, embed_dim)
        else:
            lane_features = torch.zeros(obj_features.size(0), self.lane_encoder.embedding.out_features, device=obj_data.device)
        imu_features = self.imu_encoder(imu)             # (B, embed_dim)
        # 融合特征
        combined_features = torch.cat([obj_features, lane_features, imu_features], dim=1)  # (B, embed_dim*3)
        # 扩展为 4 个 time steps 供 RNN 解码器
        combined_features = combined_features.unsqueeze(1).repeat(1, 4, 1)  # (B, 4, embed_dim*3)
        waypoints = self.decoder(combined_features)      # (B, 4, output_dim)
        return waypoints