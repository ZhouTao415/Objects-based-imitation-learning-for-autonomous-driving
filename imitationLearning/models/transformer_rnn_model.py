import torch
import torch.nn as nn

from imitationLearning.data_loader.data_loader import DrivingDataset

# Transformer Encoder: Object Features(varying length)
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, L, input_dim)
        x = self.embedding(x)  # (B, L, embed_dim)
        x = x.transpose(0, 1)  # (L, B, embed_dim)
        x = self.transformer(x)  # (L, B, embed_dim)
        x = x.transpose(0, 1)  # (B, L, embed_dim)
        return x.mean(dim = 1) # (B, embed_dim)

# MLP Encoder: Lanes and IMU Features(fixed length or padding)
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

# RNN Decoder: Generate the 4 2D waypoints
class WaypointsDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WaypointsDecoder, self).__init__()
        # self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        rnn_out, _ = self.rnn(x) # (B, seq_len, hidden_dim)
        return self.fc(rnn_out) # (B, seq_len, output_dim)

# Transformer-RNN Model: 
# Transformer processes Objects, MLP processes Lanes & IMU, and RNN generates Waypoints
class TransformerRNNModel(nn.Module):
    def __init__(self, obj_dim, lane_dim, imu_dim, 
                 embed_dim, num_heads, ff_dim, num_layers, 
                 hidden_dim, output_dim):
        super(TransformerRNNModel, self).__init__()
        self.obj_encoder = TransformerEncoder(obj_dim, embed_dim, num_heads, ff_dim, num_layers)
        self.lane_encoder = MLPEncoder(lane_dim, embed_dim)
        self.imu_encoder = MLPEncoder(imu_dim, embed_dim)
        
        self.waypoints_decoder = WaypointsDecoder(embed_dim, hidden_dim, output_dim)
        
    def forward(self, obj_data, lanes, imu):
        # obj_data: (B, L_obj, obj_dim)
        # lanes: (B, L_lane, lane_feat_dim) - first average or pool to (B, lane_feat_dim)
        # imu: (B, imu_dim) - first average or pool to (B, imu_dim)
        obj_features = self.obj_encoder(obj_data)  
        
        if lanes.size(1) > 0:
            lane_features = lanes.mean(dim=1)               # (B, lane_feat_dim) →  MLPEncoder input
        else:
            lane_features = torch.zeros(obj_features.size(0), self.lane_encoder.mlp[0].in_features, device=obj_data.device)
        lane_features = self.lane_encoder(lane_features)
        imu_features = self.imu_encoder(imu)
        combined_features = torch.cat([obj_features, lane_features, imu_features], dim=1) # (B, embed_dim*3)
        # Copy to 4 time steps for RNN decoder: (B, 4, embed_dim*3)
        combined_features = combined_features.unsqueeze(1).repeat(1, 4, 1)
        waypoiints = self.waypoints_decoder(combined_features)
        return waypoiints