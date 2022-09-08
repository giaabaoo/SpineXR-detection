import pytorch_lightning as pl
import torch.nn as nn
import csv
import os
import torch

class HistoryLogger(pl.callbacks.Callback):
    def __init__(self, dir = "history_cls.csv"):
        self.dir = dir
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "loss_epoch" in metrics.keys():
            logs = {"epoch": trainer.current_epoch}
            keys = ["loss_epoch", "train_acc_epoch", "train_f1_epoch", "test_loss","test_acc", "test_f1"]
            for key in keys:
                logs[key] = metrics[key].item()
            header = list(logs.keys())
            isFile = os.path.isfile(self.dir)
            with open(self.dir, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if not isFile:
                    writer.writeheader()
                writer.writerow(logs) 
        else:
            pass

def accuracy(y_true, y_pred):
    y_pred = torch.argmax(y_pred, axis=1, keepdim=True)
    acc = y_pred == y_true
    return acc.sum() / y_true.size(0)
def f1_score(y_true, y_pred, smooth = 1e-4):
    y_pred = torch.argmax(y_pred, axis=1, keepdim=True)
    TP = torch.sum(y_true * y_pred)
    FN = torch.sum(y_true * (1-y_pred))
    FP = torch.sum((1-y_true) * y_pred)
    precision = (TP + smooth)/(TP+FP+smooth)
    recall = (TP + smooth) /(TP+FN+smooth)
    return 2 * (precision * recall) / (precision + recall)