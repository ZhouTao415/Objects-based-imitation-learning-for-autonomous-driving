import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from models.model import WaypointPredictor
from utils.data_loader import DrivingDataset, collate_fn

def save_predictions(model, data_root, device, save_dir="predictions"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建无shuffle的数据加载器
    dataset = DrivingDataset(data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,  # 保持原始时间序列顺序
        collate_fn=collate_fn
    )
    
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            objects = batch['objects'].to(device)
            lanes = batch['lanes'].to(device)
            imu = batch['imu'].to(device)
            objects_mask = batch['objects_mask'].to(device)
            lanes_mask = batch['lanes_mask'].to(device)
            
            outputs = model(objects, lanes, imu, objects_mask, lanes_mask)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch['waypoints'].numpy())
    
    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    preds_file = f"pred_waypoints_{timestamp}.npy"
    targets_file = f"true_waypoints_{timestamp}.npy"
    
    np.save(os.path.join(save_dir, preds_file), preds_np)
    np.save(os.path.join(save_dir, targets_file), targets_np)
    print(f"Predictions saved to {os.path.join(save_dir, preds_file)}")
    print(f"Ground truth saved to {os.path.join(save_dir, targets_file)}")

def train(data_root, batch_size=8, num_epochs=1000, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置输出目录，确保存在
    os.makedirs("output", exist_ok=True)
    
    dataset = DrivingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = WaypointPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
    print("Model saved to output/waypoint_predictor.pth")
    
    return model  # 返回模型以便后续调用保存预测函数

if __name__ == '__main__':
    data_root = "./data"  # 根据实际数据路径修改
    model = train(data_root)
    
    # 如果希望训练结束后保存预测结果，可以调用 save_predictions：
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_predictions(model, data_root, device, save_dir="predictions")
