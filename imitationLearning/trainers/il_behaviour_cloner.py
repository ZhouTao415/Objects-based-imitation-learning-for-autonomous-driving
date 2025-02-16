import os
import torch
import torch.nn as nn

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

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_meter = AverageMeter()
            for batch in self.train_loader:
                obj = batch['objects'].to(self.config["device"])
                lanes = batch['lanes'].to(self.config["device"])
                imu = batch['imu'].to(self.config["device"])
                waypoints = batch['waypoints'].to(self.config["device"])

                self.optimizer.zero_grad()
                output = self.model(obj, lanes, imu)
                loss = self.criterion(output, waypoints)
                loss.backward()
                self.optimizer.step()
                train_meter.update(loss.item(), n=obj.size(0))
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_meter.avg:.4f}")

            val_loss = self.validate()
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.config["checkpoint_path"])
                print("Model saved!")

    def validate(self):
        self.model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in self.valid_loader:
                obj = batch['objects'].to(self.config["device"])
                lanes = batch['lanes'].to(self.config["device"])
                imu = batch['imu'].to(self.config["device"])
                waypoints = batch['waypoints'].to(self.config["device"])
                output = self.model(obj, lanes, imu)
                loss = self.criterion(output, waypoints)
                val_meter.update(loss.item(), n=obj.size(0))
        print(f"Validation Loss: {val_meter.avg:.4f}")
        return val_meter.avg
