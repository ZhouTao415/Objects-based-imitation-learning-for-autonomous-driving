import torch
import torch.nn as nn

# Transformer Encoder, used to process variable-length Objects data
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Use batch_first=True, directly pass in (B, L, embed_dim)
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
        return x.mean(dim=1)             # Pool to (B, embed_dim)

# MLP Encoder, used to process fixed input IMU data
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

# Lane Encoder: Use Transformer to aggregate lane data
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
        # mask: (B, L_lane) boolean, True indicates valid data
        x = self.embedding(x)         # (B, L_lane, embed_dim)
        # Transformer src_key_padding_mask requires True to indicate items to be ignored (padding),
        # so pass in ~mask (assuming mask True indicates valid data)
        key_padding_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, L_lane, embed_dim)
        return x.mean(dim=1)          # (B, embed_dim)

# Waypoints Decoder: Use RNN to generate 4 future 2D trajectory points step by step
class WaypointsDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WaypointsDecoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (B, seq_len, input_dim), here seq_len = 4
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out)  # (B, seq_len, output_dim)

# Comprehensive Model: Transformer processes Objects, LaneEncoder processes Lanes, MLP processes IMU,
# then features are fused and RNN generates Waypoints
class TransformerRNNModel(nn.Module):
    def __init__(self, obj_dim, lane_dim, imu_dim, 
                 embed_dim, num_heads, ff_dim, num_layers, 
                 hidden_dim, output_dim):
        super(TransformerRNNModel, self).__init__()
        self.obj_encoder = TransformerEncoder(obj_dim, embed_dim, num_heads, ff_dim, num_layers)
        self.lane_encoder = LaneEncoder(lane_dim, embed_dim)
        self.imu_encoder = MLPEncoder(imu_dim, embed_dim)
        # Simply concatenate the three parts of features
        self.decoder = WaypointsDecoder(embed_dim * 3, hidden_dim, output_dim)

    def forward(self, obj_data, lanes, lane_mask, imu):
        # obj_data: (B, L_obj, obj_dim)
        # lanes: (B, L_lane, lane_dim)
        # lane_mask: (B, L_lane) boolean
        # imu: (B, imu_dim)
        obj_features = self.obj_encoder(obj_data)      # (B, embed_dim)
        # Use LaneEncoder for lanes; if no lane data, fill with 0
        if lanes.size(1) > 0:
            lane_features = self.lane_encoder(lanes, lane_mask)  # (B, embed_dim)
        else:
            lane_features = torch.zeros(obj_features.size(0), self.lane_encoder.embedding.out_features, device=obj_data.device)
        imu_features = self.imu_encoder(imu)             # (B, embed_dim)
        # Fuse features
        combined_features = torch.cat([obj_features, lane_features, imu_features], dim=1)  # (B, embed_dim*3)
        # Expand to 4 time steps for RNN decoder
        combined_features = combined_features.unsqueeze(1).repeat(1, 4, 1)  # (B, 4, embed_dim*3)
        waypoints = self.decoder(combined_features)      # (B, 4, output_dim)
        return waypoints