import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import WaypointPredictor
from utils.data_loader import DrivingDataset, collate_fn

def train(data_root, batch_size=8, num_epochs=200, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建数据集与 DataLoader
    dataset = DrivingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 初始化模型
    model = WaypointPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            objects = batch['objects'].to(device)
            lanes = batch['lanes'].to(device)
            imu = batch['imu'].to(device)
            objects_mask = batch['objects_mask'].to(device)
            lanes_mask = batch['lanes_mask'].to(device)
            waypoints = batch['waypoints'].to(device)
            
            optimizer.zero_grad()
            outputs = model(objects, lanes, imu, objects_mask, lanes_mask)
            loss = criterion(outputs, waypoints)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "output/waypoint_predictor.pth")
    
if __name__ == '__main__':
    data_root = "./data"  # 请根据实际数据路径修改
    train(data_root)