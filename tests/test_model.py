import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from imitationLearning.data_loader.data_loader import DrivingDataset, ref_collate_fn
from imitationLearning.models.transformer_rnn_model import TransformerRNNModel
from imitationLearning.utils.utils import make_abs_path, load_config
from imitationLearning.utils.visualization.visualize_output import visualize_model_output

def load_model(checkpoint_path, config):
    """Load the trained model from a checkpoint."""
    model = TransformerRNNModel(
        obj_dim=config["obj_dim"],
        lane_dim=config["lane_dim"],
        imu_dim=config["imu_dim"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        output_dim=2,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"], weights_only=True))
    model.to(config["device"])
    model.eval()
    return model

def test_model(model, dataloader, device):
    """Run inference on test data and compute metrics."""
    waypoints_predicted = []
    waypoints_ground_truth = []
    
    with torch.no_grad():
        for batch in dataloader:
            obj = batch['objects'].to(device)
            lanes = batch['lanes'].to(device)
            lane_mask = batch['lane_mask'].to(device)
            imu = batch['imu'].to(device)
            waypoints = batch['waypoints'].to(device)
            
            output = model(obj, lanes, lane_mask, imu)
            waypoints_predicted.append(output.cpu().numpy())
            waypoints_ground_truth.append(waypoints.cpu().numpy())
    
    # Convert lists to NumPy arrays
    waypoints_predicted = np.concatenate(waypoints_predicted, axis=0)
    waypoints_ground_truth = np.concatenate(waypoints_ground_truth, axis=0)

    # Compute error metrics
    mse = np.mean((waypoints_predicted - waypoints_ground_truth) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(waypoints_predicted - waypoints_ground_truth))

    print(f"Test Results - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return waypoints_predicted, waypoints_ground_truth

def main():
    """Main function to run testing."""
    config = load_config("configs/model.yaml")
    device = config["device"]
    
    # Load test dataset
    # test_root = make_abs_path(__file__, "../data/test")  # Make sure test dataset exists
    test_root = make_abs_path(__file__, "../data")  # Make sure test dataset exists
    test_dataset = DrivingDataset(test_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=ref_collate_fn)

    # Load trained model
    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = load_model(checkpoint_path, config)

    # Run inference and compute error metrics
    waypoints_predicted, waypoints_ground_truth = test_model(model, test_loader, device)
    
    # 调用 visualize_output.py 中的可视化函数展示预测结果和真实轨迹
    visualize_model_output(model, test_loader, device)

if __name__ == "__main__":
    main()
