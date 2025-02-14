import torch
from torch.utils.data import DataLoader
from models.model import WaypointPredictor
from utils.data_loader import DrivingDataset, collate_fn
import torch.optim as optim
import torch.nn as nn

def train(data_root, batch_size=8, num_epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    dataset = DrivingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Model
    model = WaypointPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
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
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    data_root = 'path/to/data'  # Update this path
    train(data_root)