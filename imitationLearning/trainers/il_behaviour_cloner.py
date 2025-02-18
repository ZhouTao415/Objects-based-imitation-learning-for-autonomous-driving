import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

class BehaviourCloner:
    def __init__(self, config, model, train_loader, valid_loader, criterion, optimizer):
        self.config = config
        self.model = model.to(config["device"])
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.best_loss = float("inf")
        self.epochs = config["epochs"]

        # Used to record the training and validation loss of each epoch for subsequent plotting
        self.epoch_losses = []
        self.val_epoch_losses = []
        # Path to save the loss curve (can be modified as needed)
        plots_path = config.get("plots_path", "./output")  # Default to `output` directory
        os.makedirs(plots_path, exist_ok=True)  # Ensure `output` directory exists
        self.loss_plot_path = os.path.join(plots_path, "loss.png")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_meter = AverageMeter()
            for batch in self.train_loader:
                obj = batch['objects'].to(self.config["device"])
                lanes = batch['lanes'].to(self.config["device"])
                lane_mask = batch['lane_mask'].to(self.config["device"])
                imu = batch['imu'].to(self.config["device"])
                waypoints = batch['waypoints'].to(self.config["device"])

                self.optimizer.zero_grad()
                output = self.model(obj, lanes, lane_mask, imu)
                loss = self.criterion(output, waypoints)
                loss.backward()
                self.optimizer.step()
                train_meter.update(loss.item(), n=obj.size(0))
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_meter.avg:.4f}")
            self.epoch_losses.append(train_meter.avg)

            val_loss = self.validate()
            self.val_epoch_losses.append(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.config["checkpoint_path"])
                print("Model saved!")

        # Plot the loss curve after training
        self.plot_loss()

    def validate(self):
        self.model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in self.valid_loader:
                obj = batch['objects'].to(self.config["device"])
                lanes = batch['lanes'].to(self.config["device"])
                lane_mask = batch['lane_mask'].to(self.config["device"])
                imu = batch['imu'].to(self.config["device"])
                waypoints = batch['waypoints'].to(self.config["device"])
                output = self.model(obj, lanes, lane_mask, imu)
                loss = self.criterion(output, waypoints)
                val_meter.update(loss.item(), n=obj.size(0))
        print(f"Validation Loss: {val_meter.avg:.4f}")
        return val_meter.avg
    
    def evaluate(self, test_loader):
        """Evaluate the model on the test set and calculate error metrics (ADE, FDE)"""
        self.model.eval()
        waypoints_predicted = []
        waypoints_ground_truth = []
        
        with torch.no_grad():
            for batch in test_loader:
                obj = batch['objects'].to(self.config["device"])
                lanes = batch['lanes'].to(self.config["device"])
                lane_mask = batch['lane_mask'].to(self.config["device"])
                imu = batch['imu'].to(self.config["device"])
                waypoints = batch['waypoints'].to(self.config["device"])

                output = self.model(obj, lanes, lane_mask, imu)
                waypoints_predicted.append(output.cpu().numpy())
                waypoints_ground_truth.append(waypoints.cpu().numpy())
        
        # Concatenate the results of all batches to get an array of shape (N, T, 2)
        # where T represents the number of time steps (e.g., 4 time steps)
        waypoints_predicted = np.concatenate(waypoints_predicted, axis=0)
        waypoints_ground_truth = np.concatenate(waypoints_ground_truth, axis=0)
        
        
        # Calculate the Euclidean distance error for each time step, shape = (N, T)
        errors = np.linalg.norm(waypoints_predicted - waypoints_ground_truth, axis=2)
        
        # Average Displacement Error (ADE): the average of all time step errors
        ade = np.mean(errors)
        
        # Final Displacement Error (FDE): the error at the last time step for each sample, then averaged over all samples
        fde = np.mean(np.linalg.norm(waypoints_predicted[:, -1, :] - waypoints_ground_truth[:, -1, :], axis=1))
        
        # print(f"Test Evaluation - ADE: {ade:.4f}, FDE: {fde:.4f}")
        return ade, fde


    def plot_loss(self):
        epochs = np.arange(1, self.epochs + 1)
        plt.figure()
        plt.plot(epochs, self.epoch_losses, label="Train Loss")
        plt.plot(epochs, self.val_epoch_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.loss_plot_path)
        plt.show()