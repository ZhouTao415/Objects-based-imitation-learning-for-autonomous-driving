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

        # 用于记录每个 epoch 的训练和验证 loss，用于后续绘图
        self.epoch_losses = []
        self.val_epoch_losses = []
        # 保存 loss 曲线的路径（可根据需要修改）
        plots_path = config.get("plots_path", "./output")  # 默认为 `output` 目录
        os.makedirs(plots_path, exist_ok=True)  # 确保 `output` 目录存在
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

        # 训练结束后进行 loss 曲线绘制
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
        """在测试集上评估模型，并计算误差指标（MSE, RMSE, MAE）"""
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
        
        # 将列表拼接成 NumPy 数组
        waypoints_predicted = np.concatenate(waypoints_predicted, axis=0)
        waypoints_ground_truth = np.concatenate(waypoints_ground_truth, axis=0)
        
        # 计算误差指标
        mse = np.mean((waypoints_predicted - waypoints_ground_truth) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(waypoints_predicted - waypoints_ground_truth))
        
        print(f"Test Evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return mse, rmse, mae


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
